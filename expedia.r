# Kaggle competition expedia-hotel-recommendations

# NOTE: I won't be using my usual pipeline because this dataset is too big and I've got too little 
# time...

if (.Platform$OS.type == 'windows') {
  Sys.setenv(JAVA_HOME = 'D:\\Program Files\\Java\\jdk1.7.0_79')
}
options(java.parameters = '-Xmx16g')

library (data.table)
library (skaggle)

# Configuration
# ==================================================================================================

config = create.config('MAP@5', mode = 'single', layer = 0)

config$do.load       = T
config$do.stuff      = T
config$do.submit     = T

config$submt.id = 3
config$ref.submt.id = 2

#
# Training parameters
#

config$holdout.validation = T

# Submission
# ==================================================================================================

generate.submission = function(preds) {
  cat(date(), 'Generating submission\n')

  submission = data.frame(id = config$dte$id, hotel_cluster = apply(preds, 1, paste, collapse = ' '))
  readr::write_csv(submission, paste0('sbmt-', config$submt.id, '.csv'))
  zip(paste0('sbmt-', config$submt.id, '.zip'), paste0('sbmt-', config$submt.id, '.csv'))
  
  ref.sbmt = readr::read_csv(paste0('sbmt-', config$ref.submt.id, '.csv'))
  names(ref.sbmt)[2] = paste0('ref.', names(ref.sbmt)[2])
  ref.sbmt = ref.sbmt[order(ref.sbmt$id), ]
  ref.preds = strsplit(ref.sbmt$ref.hotel_cluster, ' ', fixed = T)
  ref.preds1 = as.numeric(unlist(lapply(ref.preds, function(x) x[[1]])))
  ref.preds1[is.na(ref.preds1)] = -1 # public scripts have some NA predictions...
  preds1 = preds[, 1]
  
  cat('Sanity check: first predictions in new and ref match', mean(preds1 == ref.preds1), 'of the time\n')
}

# Load data
# ==================================================================================================

prepare.data = function() {
  # Do elementry preprocessing and store the data as RData for a bit faster loading
  # TODO: try out feather for this when it's released
  
  train = fread(paste0(config$tmp.dir, '/train.csv'), header = T)
  test  = fread(paste0(config$tmp.dir, '/test.csv' ), header = T)
  
  # => there are 37,670,293 train samples (but only 3000693 are bookings)
  # => there are 2,528,243 test samples (all bookings)
  
  train[, date_time := as.IDate(date_time)]
  test [, date_time := as.IDate(date_time)]
  
  save(train, test, file = paste0(config$tmp.dir, '/raw.RData'))
}

load.data = function() {
  cat(date(), 'Loading data\n')

  # This takes about 3 min on my old PC!
  load(paste0(config$tmp.dir, '/raw.RData')) # => train, test

  # TODO: maybe I want to do sliding window validation

  if (config$holdout.validation) {
    cat(date(), 'NOTE: setting aside 1/3 of the training data for validation\n')
    first.valid.date = as.IDate('2014-01-01')
    valid = train[is_booking == 1 & date_time >= first.valid.date]
    train = train[date_time < first.valid.date]
    # FIXME the train/test split only includes users with events in both sets, so I probably want to
    # exclude uses with events only in one of valid/new train
    valid[, id := -(1:nrow(valid))]
  }
  
  config$dtr <<- train
  config$dte <<- test
  if (config$holdout.validation) {
    config$dva <<- valid
  }
  
  gc()
}

# Training
# ==================================================================================================

train.pred0 = function() {
  # Constant model: most common bookings overall
  # NOTE: this should give a pub LB score of 0.05949 at least (I'm not sure if they counted all 
  # events or just bookings, or what)

  cat(date(), 'Training constant model\n')

  # TODO: Can tune this on the validation set, but it's possible the optimal value is different for
  # the test set, and in any case, the validation score will be optimistic
  w1 = 20
  
  hotel_cluster_count = config$dtr[, .(w = .N + (w1 - 1) * sum(is_booking)), by = .(hotel_cluster)] 
  hotel_cluster_count = hotel_cluster_count[order(w, decreasing = T)[1:5], ]
  
  if (config$holdout.validation) {
    preds = matrix(rep(hotel_cluster_count$hotel_cluster, each = nrow(config$dva)), nrow = nrow(config$dva))
    map5 = eval.map5.core(preds, config$dva$hotel_cluster)
    cat(date(), 'Validation MAP@5 =', map5, '\n')
    # => 0.0719
  }
}

train.pred1a = function() {
  # Initial naive model posted on the forums: most popular cluster per destination
  # This is supposed to achieve around 0.30 on the pub LB.
  
  # What are the overall most common clusters? These will help us fill in predictions for 
  # destinations with few training events
  w1 = 20
  hotel_cluster_count = config$dtr[, .(w = .N + (w1 - 1) * sum(is_booking)), by = .(hotel_cluster)] 
  hotel_cluster_count = hotel_cluster_count[order(w, decreasing = T)[1:5], ]
  
  # Count occurances of hotel clusters per destination
  w2 = 5
  dest_id_hotel_cluster_count = config$dtr[, .(w = .N + (w2 - 1) * sum(is_booking)), by = .(srch_destination_id, hotel_cluster)]

  # Get the 5 most common clusters per destination. Complement with overall most common ones.
  top_five = function(hc, w) {
    hc_sorted = c(hc[order(w, decreasing = T)], hotel_cluster_count[!(hotel_cluster %in% hc), hotel_cluster])
    hc_sorted[1:5]
  }
  dest_top_five = dest_id_hotel_cluster_count[, .(hotel_cluster = top_five(hotel_cluster, w)), by = srch_destination_id]
  dest_top_five$serial = paste0('hotel_cluster', 1:5)
  dest_top_five = dcast(dest_top_five, srch_destination_id ~ serial, value.var = 'hotel_cluster')
  
  if (config$holdout.validation) {
    dd = merge(config$dva[, .(srch_destination_id, hotel_cluster)], dest_top_five, by = 'srch_destination_id', all.x = T)
    # Some destinations in the valid/test data don't appear in the train data. Fill those in
    dd[is.na(hotel_cluster1), c('hotel_cluster1', 'hotel_cluster2', 'hotel_cluster3', 'hotel_cluster4', 'hotel_cluster5') := as.list(hotel_cluster_count$hotel_cluster)]
    map5 = eval.map5.core(as.matrix(dd[, .(hotel_cluster1, hotel_cluster2, hotel_cluster3, hotel_cluster4, hotel_cluster5)]), dd$hotel_cluster)
    cat(date(), 'Validation MAP@5 =', map5, '\n')
    # => 0.3066
  }
}

train.pred1b = function() {
  # Similar to the above: most common cluster per market

  # What are the overall most common clusters? These will help us fill in predictions for 
  # markets with few training events
  w1 = 20
  hotel_cluster_count = config$dtr[, .(w = .N + (w1 - 1) * sum(is_booking)), by = .(hotel_cluster)] 
  hotel_cluster_count = hotel_cluster_count[order(w, decreasing = T)[1:5], ]
  
  # Count occurances of hotel cluster per market
  w2 = 5
  market_hotel_cluster_count = config$dtr[, .(w = .N + (w2 - 1) * sum(is_booking)), by = .(hotel_market, hotel_cluster)]
  
  # Get the 5 most common clusters per market. Complement with overall most common ones.
  top_five = function(hc, w) {
    hc_sorted = c(hc[order(w, decreasing = T)], hotel_cluster_count[!(hotel_cluster %in% hc), hotel_cluster])
    hc_sorted[1:5]
  }
  dest_top_five = market_hotel_cluster_count[, .(hotel_cluster = top_five(hotel_cluster, w)), by = hotel_market]
  dest_top_five$serial = paste0('hotel_cluster', 1:5)
  dest_top_five = dcast(dest_top_five, hotel_market ~ serial, value.var = 'hotel_cluster')
  
  if (config$holdout.validation) {
    dd = merge(config$dva[, .(hotel_market, hotel_cluster)], dest_top_five, by = 'hotel_market', all.x = T)
    # Some destinations in the valid/test data don't appear in the train data. Fill those in
    dd[is.na(hotel_cluster1), c('hotel_cluster1', 'hotel_cluster2', 'hotel_cluster3', 'hotel_cluster4', 'hotel_cluster5') := as.list(hotel_cluster_count$hotel_cluster)]
    map5 = eval.map5.core(as.matrix(dd[, .(hotel_cluster1, hotel_cluster2, hotel_cluster3, hotel_cluster4, hotel_cluster5)]), dd$hotel_cluster)
    cat(date(), 'Validation MAP@5 =', map5, '\n')
    # => 0.2487
  }
}

train.pred1c = function() {
  # Similar to the above: regularized most common cluster per XXX (trying various fields/interactions)
  # Current experiment: (hotel_market, srch_destination_id)
  # Regularization via bayesian mean with a fixed prior.

  # What are the overall most common clusters? These will help us fill in predictions for 
  # markets with few training events
  w1 = 20
  hotel_cluster_count = config$dtr[, .(w = .N + (w1 - 1) * sum(is_booking)), by = .(hotel_cluster)] 
  hotel_cluster_count = hotel_cluster_count[order(w, decreasing = T)[1:5], ]
  
  # Estimate the probability of hotel cluster per XXX
  w2 = 5 # FIXME tune
  w3 = 5 # FIXME tune
  dd = config$dtr[, .(is_booking, hotel_market, srch_destination_id, hotel_cluster)]
  dd1 = dd[, .(nxy = .N + (w3 - 1) * sum(is_booking)), by = .(hotel_market, srch_destination_id, hotel_cluster)]
  dd2 = dd[, .(nx  = .N + (w2 - 1) * sum(is_booking)), by = .(hotel_market, srch_destination_id)]
  dd3 = dd[, .(ny  = .N + (w1 - 1) * sum(is_booking)), by = .(hotel_cluster)]
  prior.n = 200 # this doesn't seem to matter almost
  prior.w = prior.n / nrow(dd)
  hc.probs = merge(dd1, dd2, by = c('hotel_market', 'srch_destination_id'))
  hc.probs = merge(hc.probs, dd3, by = 'hotel_cluster')
  hc.probs[, p := (nxy + prior.w * ny) / (nx + prior.n)]

  # Get the 5 most probable clusters per XXX. Complement with overall most common ones.
  top_five = function(hc, p) {
    hc_sorted = c(hc[order(p, decreasing = T)], hotel_cluster_count[!(hotel_cluster %in% hc), hotel_cluster])
    hc_sorted[1:5]
  }
  dest_top_five = hc.probs[, .(hotel_cluster = top_five(hotel_cluster, p)), by = .(hotel_market, srch_destination_id)]
  dest_top_five$serial = paste0('hotel_cluster', 1:5)
  dest_top_five = dcast(dest_top_five, hotel_market + srch_destination_id ~ serial, value.var = 'hotel_cluster')
  
  if (config$holdout.validation) {
    dd = merge(config$dva[, .(hotel_market, srch_destination_id, hotel_cluster)], dest_top_five, by = c('hotel_market', 'srch_destination_id'), all.x = T)
    # Some destinations in the valid/test data don't appear in the train data. Fill those in
    dd[is.na(hotel_cluster1), c('hotel_cluster1', 'hotel_cluster2', 'hotel_cluster3', 'hotel_cluster4', 'hotel_cluster5') := as.list(hotel_cluster_count$hotel_cluster)]
    map5 = eval.map5.core(as.matrix(dd[, .(hotel_cluster1, hotel_cluster2, hotel_cluster3, hotel_cluster4, hotel_cluster5)]), dd$hotel_cluster)
    cat(date(), 'Validation MAP@5 =', map5, '\n')
    # => see google sheet
  }
}

train.pred2 = function() {
  # Predict using the previous book/clicks of same user and same market/dest

  setkey(config$dtr, user_id, hotel_market)
  train.last.searches = unique(config$dtr, by = c('user_id', 'hotel_market'), fromLast = T)

  if (config$holdout.validation) {
    dva.match = merge(config$dva[, .(user_id, hotel_market, hotel_cluster)], train.last.searches[, .(user_id, hotel_market, hc = hotel_cluster)], by = c('user_id', 'hotel_market'))
    mean(dva.match$hc == dva.match$hotel_cluster)
  }
}

train.pred3 = function() {
  # Make use of the famous orig_destination_distance leak. Actually the leak is
  # exposed via: (user_location_country, user_location_region, user_location_city, hotel_market, orig_destination_distance)
  
  # The leak is that some hotels can be matched in train and test by their shared distance to 
  # similarly located users.The matching won't be perfect because some user locations can be inaccurate
  # and some distance values are simply missing, but for a large portion of samples in the testset
  # this supposedly works well. From this matching we can obtain the trainset cluster for those hotels. 
  # This too is not perfect because some hotels may change clusters over time, but it's pretty good.
  
  # It's going to be tough to make the most out of this leak...
  # The naive way to exploit it is to directly match distance values
  # - this ignores continent, country and market, which we may also want to match
  # - for small distances, it's obviously a local booking, so we could use that maybe
  # - this only looks at the first match, but should choose the five most frequent ones or help us model a nonzero probability for each match
  # - a public script that kind of does this achieved about 0.37 on the pub LB
  
  # It's highly probable that the winner in this comp will be the one who goes deepest with this leak...

  cat(date(), 'Training\n')
  
  # What are the overall most common clusters? These will help us fill in predictions for 
  # destinations with few training events
  w1 = 20
  hotel_cluster_count = config$dtr[, .(w = .N + (w1 - 1) * sum(is_booking)), by = .(hotel_cluster)] 
  hotel_cluster_count = hotel_cluster_count[order(w, decreasing = T)[1:5], ]
  
  # Count occurances of hotel cluster per destination
  w2 = 5
  dest_id_hotel_cluster_count = config$dtr[, .(w = .N + (w2 - 1) * sum(is_booking)), by = .(srch_destination_id, hotel_market, hotel_cluster)]
  
  # Get the 5 most common clusters per destination and market. Complement with overall most common ones.
  top_five = function(hc, w) {
    hc_sorted = c(hc[order(w, decreasing = T)], hotel_cluster_count[!(hotel_cluster %in% hc), hotel_cluster])
    hc_sorted[1:5]
  }
  dest_top_five = dest_id_hotel_cluster_count[, .(hotel_cluster = top_five(hotel_cluster, w)), by = .(srch_destination_id, hotel_market)]
  dest_top_five$serial = paste0('hotel_cluster', 1:5)
  dest_top_five = dcast(dest_top_five, srch_destination_id + hotel_market ~ serial, value.var = 'hotel_cluster')

  # Count occurances of hotel cluster per distance, and verify a match using the user and hotel loaction features
  # NOTE: I think there is no need to weight clicks differently, since they expose leaks just the same way bookings do
  dist_hotel_cluster_count = config$dtr[, .(w = .N), by = .(user_location_country, user_location_region, user_location_city, hotel_market, orig_destination_distance, hotel_cluster)]
  
  # Get the 5 most common clusters per distance. Complement with NAs.
  top_five = function(hc, w) {
    hc_sorted = hc[order(w, decreasing = T)]
    hc_sorted[1:5]
  }
  dist_top_five = dist_hotel_cluster_count[, .(hotel_cluster = top_five(hotel_cluster, w)), by = .(user_location_country, user_location_region, user_location_city, hotel_market, orig_destination_distance)]
  dist_top_five$serial = paste0('hotel_cluster', 1:5)
  dist_top_five = dcast(dist_top_five, user_location_country + user_location_region + user_location_city + hotel_market + orig_destination_distance ~ serial, value.var = 'hotel_cluster')

  pred = function(d) {
    # Start with our fallback destination_id based predictions
    dd.dest = merge(d[, .(srch_destination_id, hotel_market)], dest_top_five, by = c('srch_destination_id', 'hotel_market'), all.x = T, sort = F)
    dd.dest[is.na(hotel_cluster1), c('hotel_cluster1', 'hotel_cluster2', 'hotel_cluster3', 'hotel_cluster4', 'hotel_cluster5') := as.list(hotel_cluster_count$hotel_cluster)]
  
    # Now compute predictions for valid/test instances with matching train distances
    dd.dist = merge(d[, .(orig_destination_distance, user_location_country, user_location_region, user_location_city, hotel_market)], dist_top_five, by = c('user_location_country', 'user_location_region', 'user_location_city', 'hotel_market', 'orig_destination_distance'), all.x = T, sort = F)
    # Actually when the distance is NA, so should the predictions be
    dd.dist[is.na(orig_destination_distance), c('hotel_cluster1', 'hotel_cluster2', 'hotel_cluster3', 'hotel_cluster4', 'hotel_cluster5')] = NA
    
    # Fill in any missing predictions from the fallback
    dd = cbind(dd.dist, dd.dest[, .(fhc1 = hotel_cluster1, fhc2 = hotel_cluster2, fhc3 = hotel_cluster3, fhc4 = hotel_cluster4, fhc5 = hotel_cluster5)])
    dd[, id := .I]
    dd = dd[, .(hc = unique(na.omit(c(hotel_cluster1, hotel_cluster2, hotel_cluster3, hotel_cluster4, hotel_cluster5, fhc1, fhc2, fhc3, fhc4, fhc5)))[1:5]), by = id]
    dd[, serial := paste0('hotel_cluster', 1:5)]
    dd = dcast(dd, id ~ serial, value.var = 'hc')

    as.matrix(dd[, .(hotel_cluster1, hotel_cluster2, hotel_cluster3, hotel_cluster4, hotel_cluster5)])
  }
  
  if (config$holdout.validation) {
    cat(date(), 'Predicting on validset\n')
    preds = pred(config$dva)
    map5 = eval.map5.core(preds, config$dva$hotel_cluster)
    cat(date(), 'Validation MAP@5 =', map5, '\n')
    # => 0.4509
  }

  if (config$do.submit) {
    cat(date(), 'Predicting on testset\n')
    preds = pred(config$dte)
    generate.submission(preds)
  }
}

gen.preds.marginal = function() {
  marginal.counts = config$dtr[, .(w = .N + 19 * sum(is_booking)), by = hotel_cluster]
  marginal.counts = marginal.counts$w[order(marginal.counts$hotel_cluster)]
  marginal.preds = marginal.counts / sum(marginal.counts)
  
  if (config$holdout.validation) {
    save(marginal.preds, file = paste0(config$tmp.dir, '/marginal-preds-2014.RData'))
    
    preds = order(marginal.preds, decreasing = T)[1:5] - 1
    preds = matrix(rep(preds, each = nrow(config$dva)), ncol = 5)
    map5 = eval.map5.core(preds, config$dva$hotel_cluster)
    cat(date(), 'Validation MAP@5 =', map5, '\n')
    # => 0.0719
  }
  
  if (config$do.submit) {
    save(marginal.preds, file = paste0(config$tmp.dir, '/marginal-preds-2015.RData'))
  }
}

gen.preds.desmar = function() {
  n0 = 5 # Amazing, but it's better not to shrink?? (or almost so with a tiny effect)
  
  dd1 = config$dtr[hotel_market != 0, .(nxy = .N + 4 * sum(is_booking)), by = .(hotel_continent, hotel_country, hotel_market, srch_destination_id, hotel_cluster)]
  dd1[, nx  := sum(nxy), by = .(hotel_continent, hotel_country, hotel_market, srch_destination_id)]
  dd1[, w   := pmin(nx / n0, 1)]
  dd1[, p   := nxy / nx]
  dd1.p = dcast(dd1, hotel_continent + hotel_country + hotel_market + srch_destination_id ~ hotel_cluster, value.var = 'p', fill = 0)
  dd1 = merge(dd1.p, unique(dd1[, .(hotel_continent, hotel_country, hotel_market, srch_destination_id, w)]), by = c('hotel_continent', 'hotel_country', 'hotel_market', 'srch_destination_id'))
  rm(dd1.p)
  
  dd2 = config$dtr[hotel_market != 0, .(nxy = .N + 4 * sum(is_booking)), by = .(hotel_continent, hotel_country, hotel_market, hotel_cluster)]
  dd2[, nx  := sum(nxy), by = .(hotel_continent, hotel_country, hotel_market)]
  dd2[, w   := pmin(nx / n0, 1)]
  dd2[, p   := nxy / nx]
  dd2.p = dcast(dd2, hotel_continent + hotel_country + hotel_market ~ hotel_cluster, value.var = 'p', fill = 0)
  dd2 = merge(dd2.p, unique(dd2[, .(hotel_continent, hotel_country, hotel_market, w)]), by = c('hotel_continent', 'hotel_country', 'hotel_market'))
  rm(dd2.p)
  
  dd3 = config$dtr[hotel_market != 0, .(nxy = .N + 4 * sum(is_booking)), by = .(hotel_continent, hotel_country, hotel_cluster)]
  dd3[, nx  := sum(nxy), by = .(hotel_continent, hotel_country)]
  dd3[, w   := pmin(nx / n0, 1)]
  dd3[, p   := nxy / nx]
  dd3.p = dcast(dd3, hotel_continent + hotel_country ~ hotel_cluster, value.var = 'p', fill = 0)
  dd3 = merge(dd3.p, unique(dd3[, .(hotel_continent, hotel_country, w)]), by = c('hotel_continent', 'hotel_country'))
  rm(dd3.p)
  
  dd4 = config$dtr[hotel_market != 0, .(nxy = .N + 4 * sum(is_booking)), by = .(hotel_continent, hotel_cluster)]
  dd4[, nx  := sum(nxy), by = .(hotel_continent)]
  dd4[, w   := pmin(nx / n0, 1)]
  dd4[, p   := nxy / nx]
  dd4.p = dcast(dd4, hotel_continent ~ hotel_cluster, value.var = 'p', fill = 0)
  dd4 = merge(dd4.p, unique(dd4[, .(hotel_continent, w)]), by = c('hotel_continent'))
  rm(dd4.p)
  
  gc()
  
  get.preds = function(d) {
    p = merge(d[, .(hotel_continent)], dd4, by = c('hotel_continent'), all.x = T, sort = F)[, 1 + (1:100), with = F]
    p[is.na(p)] = 0
    p = data.matrix(p)
    
    pn = merge(d[, .(hotel_continent, hotel_country)], dd3, by = c('hotel_continent', 'hotel_country'), all.x = T, sort = F)[, 2 + (1:101), with = F]
    pn[is.na(pn)] = 0
    wn = pn$w
    pn = data.matrix(pn[, 1:100, with = F])
    p = wn * pn + (1 - wn) * p
    gc()
    
    pn = merge(d[, .(hotel_continent, hotel_country, hotel_market)], dd2, by = c('hotel_continent', 'hotel_country', 'hotel_market'), all.x = T, sort = F)[, 3 + (1:101), with = F]
    pn[is.na(pn)] = 0
    wn = pn$w
    pn = data.matrix(pn[, 1:100, with = F])
    p = wn * pn + (1 - wn) * p
    gc()

    pn = merge(d[, .(hotel_continent, hotel_country, hotel_market, srch_destination_id)], dd1, by = c('hotel_continent', 'hotel_country', 'hotel_market', 'srch_destination_id'), all.x = T, sort = F)[, 4 + (1:101), with = F]
    pn[is.na(pn)] = 0
    wn = pn$w
    pn = data.matrix(pn[, 1:100, with = F])
    p = wn * pn + (1 - wn) * p
    gc()
    
    return (p)
  }
  
  if (config$holdout.validation) {
    preds = get.preds(config$dva)
    save(preds, file = paste0(config$tmp.dir, '/desmar-preds-2014.RData'))

    preds = t(apply(preds, 1, order, decreasing = T))[, 1:5] - 1
    map5 = eval.map5.core(preds, config$dva$hotel_cluster)
    cat(date(), 'Validation MAP@5 =', map5, '\n')
    # => 0.3123
  }
  
  if (config$do.submit) {
    preds = get.preds(config$dte)
    save(preds, file = paste0(config$tmp.dir, '/desmar-preds-2015.RData'))
  }
}

gen.preds.leak = function() {
  dist.matches = config$dtr[!is.na(orig_destination_distance), .N, by = .(user_location_country, user_location_region, user_location_city, hotel_market, orig_destination_distance, hotel_cluster)]
  #dist.matches = config$dtr[!is.na(orig_destination_distance), .(N = length(unique(user_id))), by = .(user_location_country, user_location_region, user_location_city, hotel_market, orig_destination_distance, hotel_cluster)]
  dist.matches = dcast(dist.matches, user_location_country + user_location_region + user_location_city + hotel_market + orig_destination_distance ~ hotel_cluster, value.var = 'N', fill = 0)
  setnames(dist.matches, as.character(0:99), paste0('p', 0:99))
  
  if (config$holdout.validation) {
    preds = merge(config$dva[, .(user_location_country, user_location_region, user_location_city, hotel_market, orig_destination_distance)], dist.matches, by = c('user_location_country', 'user_location_region', 'user_location_city', 'hotel_market', 'orig_destination_distance'), all.x = T, sort = F)[, 5 + (1:100), with = F]
    preds[is.na(preds)] = 0
    nrmlz = rowSums(preds)
    idx = (nrmlz > 0)
    preds[idx] = preds[idx] / nrmlz[idx]
    save(preds, file = paste0(config$tmp.dir, '/leak-preds-2014.RData'))

    preds = t(apply(preds, 1, order, decreasing = T))[, 1:5] - 1
    map5 = eval.map5.core(preds, config$dva$hotel_cluster)
    cat(date(), 'Validation MAP@5 =', map5, '\n')
    # => 0.2383
  }
  
  if (config$do.submit) {
    preds = merge(config$dte[, .(user_location_country, user_location_region, user_location_city, hotel_market, orig_destination_distance)], dist.matches, by = c('user_location_country', 'user_location_region', 'user_location_city', 'hotel_market', 'orig_destination_distance'), all.x = T, sort = F)[, 5 + (1:100), with = F]
    cat(date, 'FYI: the leak covers', mean(!is.na(preds[, 1])), 'of test examples\n')
    preds[is.na(preds)] = 0
    nrmlz = rowSums(preds)
    idx = (nrmlz > 0)
    preds[idx] = preds[idx] / nrmlz[idx]
    save(preds, file = paste0(config$tmp.dir, '/leak-preds-2015.RData'))
  }
}

blend.preds = function() {
  #     marginal, global-xgb, desmar-xgb, leak
  w = c(0       , 1   , 10        , 100  )
  w = w / sum(w)
  
  if (config$holdout.validation) {
    pred.year = 2014
  } else {
    pred.year = 2015
  }
  
  # FIXME maybe I want to do geomean
  load(paste0(config$tmp.dir, '/marginal-preds-', pred.year, '.RData')) # => marginal.preds
  load(paste0(config$tmp.dir, '/global-xgb-preds-', pred.year, '.RData')) # => preds (NOTE: this is generated by global-xgb.r)
  b.preds = t(marginal.preds * w[1] + t(preds * w[2]))
  load(paste0(config$tmp.dir, '/desmar-xgb-preds-', pred.year, '.RData')) # => preds (NOTE: this is generated by desmar-xgb.r)
  b.preds = b.preds + data.matrix(preds) * w[3]
  load(paste0(config$tmp.dir, '/leak-preds-', pred.year, '.RData')) # => preds
  b.preds = b.preds + data.matrix(preds) * w[4]
  preds = t(apply(b.preds, 1, order, decreasing = T))[, 1:5] - 1

  if (config$holdout.validation) {
    map5 = eval.map5.core(preds, config$dva$hotel_cluster)
    cat(date(), 'Validation MAP@5 =', map5, '\n')
    # => 0.4574
  }
  
  if (config$do.submit) {
    generate.submission(preds)
  }
}

# Do stuff
# ==================================================================================================

if (config$mode == 'single') {
  cat(date(), 'Starting single mode\n')
  
  if (config$do.load) {
    load.data()
  }
  
  if (config$do.stuff) {
    gen.preds.marginal()
    gc()
    gen.preds.desmar()
    gc()
    gen.preds.leak()
    gc()
    blend.preds()
    gc()
  }
}

cat(date(), 'Done.\n')
