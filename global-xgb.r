# XGB model for the entire data
# - one against the rest
# - direct multiclass

library (Matrix)
library (data.table)
library (xgboost)

source('../utils.r')

do.preprocess     = T
debug.one.vs.rest = F
do.one.vs.rest    = T
do.multiclass     = F

do.validation = F

#tmpdir = 'c:/TEMP/kaggle/expedia-hotel-recommendations'
tmpdir = '~/Scratch/Kaggle/expedia-hotel-recommendations'

if (do.preprocess) {
  cat(date(), 'Preprocessing\n')
  
  #train = fread('train.csv'), header = T)
  #test  = fread('test.csv' ), header = T)
  #save(train, test, file = paste0(tmp.dir, '/raw.RData'))
  load(paste0(tmp.dir, '/raw.RData')) # => train, test
  
  # FIXME: make use of clicks too on an appropriate infrastructure (it's too big for my individual machines)
  train = train[is_booking == 1, ]
  gc()

  train[, is_booking := NULL]
  train[, cnt := NULL] # maybe dropping some info here if nonzero, but we don't have this for test
  test[, id := NULL]
  test[, hotel_cluster := NA]
  dat = rbind(train, test)
  rm(train, test)
  gc()

  # Dates
  dat[, date_time := as.IDate(date_time)]
  dat[, year  := year (date_time)]
  dat[, month := month(date_time)]
  dat[, srch_ci := as.IDate(srch_ci)]
  dat[, srch_co := as.IDate(srch_co)]
  dat[, srch_ci.month := month(srch_ci)]
  dat[, srch_ci.dow := wday(srch_ci)]
  dat[, srch_co.dow := wday(srch_co)]
  dat[, days.to.ci := as.integer(srch_ci - date_time)]
  dat[days.to.ci < 0, days.to.ci := NA] # ??
  dat[, nr.nights := as.integer(srch_co - srch_ci)]
  dat[nr.nights < 0, nr.nights := NA] # ??
  
  # Low cardinality factors - to be OHE
  dat[, site_name                := as.factor(site_name               )]
  dat[, posa_continent           := as.factor(posa_continent          )]
  dat[, channel                  := as.factor(channel                 )]
  dat[, srch_destination_type_id := as.factor(srch_destination_type_id)]
  dat[, hotel_continent          := as.factor(hotel_continent         )]
  
  # High cardinality factors - counts over entire 2013-2015 period
  # FIXME is this useful? do we want to sum over different periods, compute differences, ...?
  dat[, cnt.user_location_city    := .N, by = user_location_city   ]
  dat[, cnt.user_location_country := .N, by = user_location_country]
  dat[, cnt.user_id               := .N, by = user_id              ]
  dat[, cnt.srch_destination_id   := .N, by = srch_destination_id  ]
  dat[, cnt.hotel_market          := .N, by = hotel_market         ]
  
  # Add an indicator for whether the user and hotel countries (probably) match
  load('closest-countries.RData') # => closest.countries
  setnames(closest.countries, c('user_location_country', 'country.match'))
  dat = merge(dat, closest.countries, by = 'user_location_country', all.x = T, sort = F)
  dat[, country.match := as.integer(hotel_country == country.match)]

  #
  # Finalize data for modeling
  #
  
  cat(date(), 'Finalizing modeling data\n')

  drop.features = c('site_name', 'posa_continent', 'srch_ci', 'srch_co')
  meta.feature.names = c('date_time', 'user_id', 'user_location_country', 'user_location_region', 'user_location_city', 'srch_destination_id', 'hotel_country', 'hotel_market', 'hotel_cluster')
  dat.model = dat[, !(names(dat) %in% c(drop.features, meta.feature.names)), with = F]  
  dat.meta = dat[, .(date_time, user_id, hotel_cluster)]
  rm(dat); gc()

  # Take care of NAs and OHE
  for (f in names(dat.model)) {
    if (is.factor(dat.model[[f]])) {
      if (any(is.na(dat.model[[f]]))) {
        x = as.character(dat.model[[f]])
        x[is.na(x)] = 'NA'
        dat.model[, f] = as.factor(x)
      }
    } else {
      dat.model[is.na(dat.model[[f]]), f] = -999
    }
  }
  dat.model = sparse.model.matrix(~ . - 1, dat.model)
  
  # Split to actual-train, validation and test sets in a way that best reflects the train/test split
  # FIXME maybe this could be done better, and maybe I could do some sort of cross validation
  first.valid.date = as.IDate('2014-01-01')
  first.test.date  = as.IDate('2015-01-01')
  
  train.idxs = (dat.meta$date_time <  first.valid.date)
  valid.idxs = (dat.meta$date_time >= first.valid.date) & (dat.meta$date_time < first.test.date)
  test.idxs  = (dat.meta$date_time >= first.test.date )
  
  # maybe I wanna do this?
  #valid.idxs = valid.idxs & (dat.meta$user_id %in% unique(dat.meta[train.idxs, user_id])) 
  # maybe I also want to remove users not in valid from the trainset?
  
  train = dat.model[train.idxs, ]
  valid = dat.model[valid.idxs, ]
  test  = dat.model[test.idxs , ]
  rm(dat.model)
  
  train.meta = dat.meta[train.idxs, ]
  valid.meta = dat.meta[valid.idxs, ]
  test.meta  = dat.meta[test.idxs , ]
  rm(dat.meta)
  
  cat(date(), 'Saving data\n')
  save(train, train.meta, file = paste0(tmpdir, '/pp-train-data.RData'))
  save(valid, valid.meta, file = paste0(tmpdir, '/pp-valid-data.RData'))
  save(test , test.meta , file = paste0(tmpdir, '/pp-test-data.RData' ))
  gc()
} else {
  cat(date(), 'Loading previously preprocessed data\n')
  load(paste0(tmpdir, '/pp-train-data.RData')) # => train, train.meta
  load(paste0(tmpdir, '/pp-valid-data.RData')) # => valid, valid.meta
}

if (0) {
  # Problem: this takes too much memory. 
  # Ideas:
  # 1. model on this separately, and blend the results (or even use a small validation set to optimize blending weight)
  # 2. use a huge mem machine...
  
  # TODO: use CV (or CCV if needed) and fit cluster probabilities marginally on
  # each of these big categoricals (and maybe the smaller ones too). Since this
  # will add many features (100 per original feature), in the context of a 
  # one-vs-rest approach it might make more sense to add only the fit wrt the 
  # one cluster at a time.
  
  load('dest-yenc.RData') # => dest.yenc
  dat = merge(dat, dest.yenc, by = 'srch_destination_id', all.x = T, sort = F)
  rm(dest.yenc)
  gc()
  # Now I can do LOO if I want by subtracting cluster counts per example
  # FIXME maybe this will leak and I want CV/CCV (and maybe need to take time into account...)
  for (i in 0:99) {
    tmp = c(data.matrix(dat[hotel_cluster == i, paste0('dest.yenc', i), with = F])) - 1
    dat[hotel_cluster == i, paste0('dest.yenc', i) := tmp]
  }
  tmp2 = rowSums(dat[, paste0('dest.yenc', 0:99), with = F])
  idxs = !is.na(tmp2)
  for (i in 0:99) {
    tmp = c(data.matrix(dat[idxs, paste0('dest.yenc', i), with = F])) / tmp2[idxs]
    dat[idxs, paste0('dest.yenc', i) := tmp]
  }
  gc()

  # TODO: merge the destination table? dim reduced maybe?
}

if (debug.one.vs.rest) {
  # Let's look at one of the one-vs-rest problems as an example
  curr.target.cluster = 10
  
  feature.names = colnames(train)
  xtrain = xgb.DMatrix(train, label = as.integer(train.meta$hotel_cluster == curr.target.cluster))
  xvalid = xgb.DMatrix(valid, label = as.integer(valid.meta$hotel_cluster == curr.target.cluster))
  rm(train, valid)
  gc()
  
  watchlist = list(valid = xvalid, train = xtrain)
  
  yr = sum(getinfo(xtrain, 'label') == 0) / sum(getinfo(xtrain, 'label') == 1)
  
  xgb.params = list(
    #booster           = 'gbtree',
    #booster           = 'gblinear',
    #objective         = 'reg:linear',
    objective         = 'binary:logistic',
    #objective         = 'rank:pairwise',
    #eval_metric       = 'rmse',
    eval_metric       = 'error',
    nrounds           = 500,
    #eta               = 0.3,
    #max_depth         = 6,
    #min_child_weight  = 1,
    #scale_pos_weight  = 10,
    #max_delta_step    = 0,
    #gamma             = 0,
    #lambda            = 1,
    #alpha             = 0,
    #num_parallel_tree = 1,
    #subsample         = 1,
    #colsample_bytree  = 1,
    annoying = T
  )
  
  cat(date(), 'Training\n')
  
  set.seed(1234)
  
  xgb.fit = xgb.train(
    params            = xgb.params,
    nrounds           = xgb.params$nrounds,
    data              = xtrain,
    watchlist         = watchlist,
    print.every.n     = 10,
    nthread           = 8
  )
  
  preds0 = predict(xgb.fit, xvalid)
  cat('Constant model error:', mean(getinfo(xvalid, 'label')), '\n')
  cat('Constant model error:', mean(ifelse(preds0 > 0.999, 1, 0) != getinfo(xvalid, 'label')), '\n')
  
  if (0) {
    cat(date(), 'Checking importance\n')
    impo = xgb.importance(feature.names, model = xgb.fit)
    print(impo[1:50, ])
  }
  
  # => so far I've been unable to do very little with this problem...
  # It looks like, at least without the high cardinality categorical features, it
  # is extremely difficult to differentiate between cluster 0 and the other ones.
  
  # => maybe I would be better off finding an easier problem to work on? (e.g.
  # see which clusters are most affected by each feature)
}

if (do.one.vs.rest) {
  # If training a 100-class multinomial logistic regression is infeasible for this dataset size,
  # let's use the one-vs-rest heuristic: build 100 2-class models where in model k the positives are
  # only examples with class k, and remaining examples are negatives; then rank the probability 
  # scores from the 100 models to generate a prediction.

  # FIXME maybe I want to do some feature eng and tuning per problem
  feature.names = colnames(train)
  xtrain = xgb.DMatrix(train)
  xvalid = xgb.DMatrix(valid)
  rm(train, valid)
  gc()
  
  # FIXME since each one-vs-rest problem will be imbalanced, I might want to add max_delta_tree or whatever it's called
  xgb.params = list(
    #booster           = 'gbtree',
    #booster           = 'gblinear',
    #objective         = 'reg:linear',
    objective         = 'binary:logistic',
    #objective         = 'rank:pairwise',
    #eval_metric       = 'rmse',
    #eval_metric       = 'score',
    nrounds           = 500,
    #eta               = 0.03,
    #max_depth         = 6,
    #min_child_weight  = 1,
    #gamma             = 0,
    #lambda            = 0,
    #alpha             = 0,
    #num_parallel_tree = 1,
    #subsample         = 1,
    #colsample_bytree  = 0.5,
    annoying = T
  )
  
  all.preds = matrix(NA, nrow(xvalid), 100)
  
  for (k in 1:100) {
    cat(date(), 'Working on class', k, '\n')
    
    # NOTE: assuming newer version of XGB that handles NAs in the xgb.DMatrix interface by default
    setinfo(xtrain, 'label', as.integer(train.meta$hotel_cluster == k - 1))
    setinfo(xvalid, 'label', as.integer(valid.meta$hotel_cluster == k - 1))
    watchlist = list(valid = xvalid, train = xtrain)
    set.seed(1234)
    
    xgb.fit = xgb.train(
      params            = xgb.params,
      nrounds           = xgb.params$nrounds,
      data              = xtrain,
      watchlist         = watchlist,
      print.every.n     = 100,
      nthread           = 8
    )
    
    all.preds[, k] = predict(xgb.fit, xvalid)
  }
  
  preds = t(apply(all.preds, 1, order, decreasing = T))[, 1:5] - 1
  eval.map5.core = function(preds, labels) {
    succ = (preds == labels)
    w = 1 / (1:5)
    map5 = mean(succ %*% w)
  }
  map5 = eval.map5.core(preds, valid$hotel_cluster)
  cat(date(), 'Validation MAP@5 =', map5, '\n')
  # => 0.2988 ... but even this I don't understand, if all the models are useless how does it perform so well compared to the constant model?
}

if (do.multiclass) {
  feature.names = colnames(train)
  if (do.validation) {
    xtrain = xgb.DMatrix(train, label = train.meta$hotel_cluster)
    xvalid = xgb.DMatrix(valid, label = valid.meta$hotel_cluster)
    watchlist = list(valid = xvalid, train = xtrain)
  } else {
    xtrain = xgb.DMatrix(rbind(train, valid), label = c(train.meta$hotel_cluster, valid.meta$hotel_cluster))
    watchlist = NULL
  }
  rm(train, valid)
  gc()
  
  eval.map5 = function(preds, dtrain) {
    labels = getinfo(dtrain, 'label')
    preds = t(matrix(preds, ncol = length(labels)))
    preds = t(apply(preds, 1, order, decreasing = T))[, 1:5] - 1
    succ = (preds == labels)
    w = 1 / (1:5)
    map5 = mean(succ %*% w)
    return (list(metric = 'map5', value = map5))
  }
  
  xgb.params = list(
   #booster           = 'gbtree',
   #booster           = 'gblinear',
   #objective         = 'reg:linear',
   #objective         = 'binary:logistic',
   #objective         = 'multi:softmax',
    objective         = 'multi:softprob',
   #objective         = 'rank:pairwise',
   #eval_metric       = 'rmse',
   #eval_metric       = 'error',
   #eval_metric       = 'merror',
    eval_metric       = 'mlogloss',
   #eval_metric       = eval.map5,   # slow
    maximize          = F, #T,       # but need this if map5
    nrounds           = 300,
   #eta               = 0.3,
   #max_depth         = 6,
   #min_child_weight  = 1,
   #scale_pos_weight  = 10,
   #max_delta_step    = 0,
   #gamma             = 0,
   #lambda            = 1,
   #alpha             = 0,
   #num_parallel_tree = 1,
   #subsample         = 1,
   #colsample_bytree  = 1,
    num_class         = 100,
    annoying = T
  )
  
  cat(date(), 'Training\n')
  
  set.seed(1234)
  
  xgb.fit = xgb.train(
    params            = xgb.params,
    nrounds           = xgb.params$nrounds,
    maximize          = xgb.params$maximize,
    data              = xtrain,
    watchlist         = watchlist,
    print.every.n     = 10,
    nthread           = 8,
    save_period       = 100,
    save_name         = path.expand(paste0(tmpdir, '/model.xgb'))
  )
  
  if (0) {
    xgb.fit = xgb.load(path.expand(paste0(tmpdir, '/model.xgb')))
  }
  
  if (0) {
    cat(date(), 'Checking importance\n')
    impo = xgb.importance(feature.names, model = xgb.fit)
    save(impo, file = 'xgb-importance.RData')
    print(impo[1:50, ])
  }
  
  if (do.validation) {
    preds = predict(xgb.fit, xvalid)
    map5 = eval.map5(preds, xvalid)
    cat(date(), 'Validation map5 =', map5$value, '\n')
    # => 0.2929
    preds = t(matrix(preds, ncol = nrow(xvalid)))
    save(preds, file = paste0(tmpdir, '/global-xgb-preds-2014.RData'))

    load(paste0(tmpdir, '/pp-test-data.RData')) # => test, test.meta
    xtest = xgb.DMatrix(test)
    rm(test)
    gc()
    preds = t(matrix(predict(xgb.fit, xtest), ncol = nrow(xtest)))
    save(preds, file = paste0(tmpdir, '/global-xgb-preds-2015v.RData'))
  } else {
    load(paste0(tmpdir, '/pp-test-data.RData')) # => test, test.meta
    xtest = xgb.DMatrix(test)
    rm(test)
    gc()
    cat(date(), 'Predicting on test\n')
    preds = t(matrix(predict(xgb.fit, xtest), ncol = nrow(xtest)))
    save(preds, file = paste0(tmpdir, '/global-xgb-preds-2015.RData'))
    cat(date(), 'Done.\n')
  }
}
