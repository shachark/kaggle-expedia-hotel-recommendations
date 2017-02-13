# Separate XGB models per destination / market / both

library (Matrix)
library (data.table)
library (xgboost)
library (skaggle)

do.preprocess = T
do.train      = T
do.compose    = T

do.validation = T
do.reference  = F
do.autotune   = T
subset.on = 'dest' # { market, dest }
top.subset.is = 1:169 # TODO: I got stuck at subset 170 (so 1:169 is done, but in theory, the more the merrier)

first.valid.date = as.IDate('2014-01-01')
first.test.date  = as.IDate('2015-01-01')

#tmpdir = 'c:/TEMP/kaggle/expedia-hotel-recommendations'
tmpdir = '~/Scratch/Kaggle/expedia-hotel-recommendations'

if (do.preprocess) {
  cat(date(), 'Preprocessing\n')
  
  #train = fread('train.csv'), header = T)
  #test  = fread('test.csv' ), header = T)
  #save(train, test, file = paste0(tmpdir, '/raw.RData'))
  load(paste0(tmpdir, '/raw.RData')) # => train, test
  
  # TODO: make use of clicks too (probably using per example weights)
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
  dat[, site_name      := as.factor(site_name     )]
  dat[, posa_continent := as.factor(posa_continent)]
  dat[, channel        := as.factor(channel       )]
  if (0) { # these will be constant here
    dat[, srch_destination_type_id := as.factor(srch_destination_type_id)] 
    dat[, hotel_continent          := as.factor(hotel_continent         )]
  } else {
    dat[, srch_destination_type_id := NULL] 
    dat[, hotel_continent          := NULL]
  }

  # High cardinality factors - counts over entire 2013-2015 period
  # TODO do we want to sum over different periods, compute differences, ...?
  dat[, cnt.user_id               := .N, by = user_id]
  dat[, cnt.user_location_city    := .N, by = .(user_location_country, user_location_region, user_location_city)]
  dat[, cnt.user_location_region  := .N, by = .(user_location_country, user_location_region)]
  dat[, cnt.user_location_country := .N, by = user_location_country]
  dat[, cnt.srch_destination_id   := .N, by = srch_destination_id]
  dat[, cnt.hotel_market          := .N, by = hotel_market]

  # TODO: if subsetting on market rather than dest id, merge data from the destinations file
  
  # Add an indicator for whether the user and hotel countries (probably) match
  load('closest-countries.RData') # => closest.countries
  setnames(closest.countries, c('user_location_country', 'country.match'))
  dat = merge(dat, closest.countries, by = 'user_location_country', all.x = T, sort = F)
  dat[, country.match := as.integer(hotel_country == country.match)]
  rm(closest.countries)

  # Add the overall median distance of user city to hotel market 
  # FIXME: maybe actually use this to impute distance, or include it as a separate feature
  # FIXME maybe I really only need the median? it looks pretty stable
  load('city-dists.RData') # => city.dists
  setnames(city.dists, c('user_location_country', 'user_location_region', 'user_location_city', 'hotel_market', 'city.dist.q0', 'city.dist.q1', 'city.dist.q2', 'city.dist.q3', 'city.dist.q4'))
  dat = merge(dat, city.dists, by = c('user_location_country', 'user_location_region', 'user_location_city', 'hotel_market'), all.x = T, sort = F)
  rm(city.dists)
  
  if (1) {
    # High cardinality factors - hoping now we can OHE these as well
    dat[, user_location_region  := as.factor(paste0(user_location_country, '-', user_location_region))] # about 2000 levels
    dat[, user_location_country := as.factor(user_location_country)] # about 250 levels
    if (subset.on == 'market') {
      dat[, srch_destination_id := as.factor(srch_destination_id)] # up to about 500 levels per market
    }
  }
  
  cat(date(), 'Saving data\n')
  
  drop.features = c('srch_ci', 'srch_co') # 'site_name', 'posa_continent'
  meta.feature.names = c('date_time', 'user_id', 'user_location_city', 'hotel_market', 'hotel_country', 'srch_destination_id', 'hotel_cluster')
  dat.model = dat[, !(names(dat) %in% c(drop.features, meta.feature.names)), with = F]  
  dat.meta = dat[, meta.feature.names, with = F]
  rm(dat); gc()
  
  save(dat.model, dat.meta, file = paste0(tmpdir, '/pp-data.RData'))
}

if (do.train) {
  nr.top.subsets = length(top.subset.is)
  top.subset.map5s = rep(NA, nr.top.subsets)
  top.subset.ref.map5s = rep(NA, nr.top.subsets)
  
  for (top.subset.i in top.subset.is) {
    cat(date(), 'Working on subset', top.subset.i, '\n')
    
    #
    # Finalize data for modeling
    #
    
    cat(date(), 'Loading preprocessed data\n')
    load(paste0(tmpdir, '/pp-data.RData')) # => dat.model, dat.meta
    
    if (subset.on == 'market') {
      d.sub = dat.meta[, .N, by = hotel_market]
      d.sub = d.sub[order(N, decreasing = T)]
      idx = (dat.meta$hotel_market == d.sub$hotel_market[top.subset.i])
    } else if (subset.on == 'dest') {
      d.sub = dat.meta[, .N, by = srch_destination_id]
      d.sub = d.sub[order(N, decreasing = T)]
      idx = (dat.meta$srch_destination_id == d.sub$srch_destination_id[top.subset.i])
    } else {
      stop('WTF')
    }
  
    dat.meta  = dat.meta [idx]
    dat.model = dat.model[idx]
    rm(d.sub, idx)
    
    cat(date(), 'Finalizing modeling data\n')
    
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

    if (do.reference) {
      #cat(date(), 'Fitting reference model\n')
      preds = train.meta[, .(p = .N / nrow(train.meta)), by = hotel_cluster]
      preds = preds[order(p, decreasing = T)]$hotel_cluster[1:5]
      
      if (do.validation) {
        preds = matrix(preds, nrow(valid.meta), 5, byrow = T)
        succ = (preds == valid.meta$hotel_cluster)
        w = 1 / (1:5)
        map5 = mean(succ %*% w)
        #cat(date(), 'Validation MAP@5 =', map5, '\n')
        top.subset.ref.map5s[top.subset.i] = map5
      }
    }
    
    feature.names = colnames(train)
    if (do.validation) {
      xtrain = xgb.DMatrix(train, label = train.meta$hotel_cluster)
      xvalid = xgb.DMatrix(valid, label = valid.meta$hotel_cluster)
      watchlist = list(valid = xvalid, train = xtrain)
    } else {
      xtrain = xgb.DMatrix(rbind(train, valid), label = c(train.meta$hotel_cluster, valid.meta$hotel_cluster))
      watchlist = NULL
    }
    xtest = xgb.DMatrix(test)
    rm(train, valid, test)
    gc()
    
    #
    # Setup hyperparams / tune (using standard CV on the training set)
    #

    eval.map5 = function(preds, dtrain) {
      labels = getinfo(dtrain, 'label')
      preds = t(matrix(preds, ncol = length(labels)))
      preds = t(apply(preds, 1, order, decreasing = T))[, 1:5] - 1
      succ = (preds == labels)
      w = 1 / (1:5)
      map5 = mean(succ %*% w)
      return (list(metric = 'map5', value = map5))
    }
    
    #stop('DEBUGGING')
    
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
      nrounds           = 200,
      eta               = 0.1, #0.05,
      max_depth         = 8,
      #min_child_weight  = 1,
      #scale_pos_weight  = 10,
      #max_delta_step    = 0,
      #gamma             = 0,
      #lambda            = 0,
      #alpha             = 0,
      #num_parallel_tree = 1,
      subsample         = 0.9,
      colsample_bytree  = 0.7,
      num_class         = 100,
      
      tune.nrounds          = 500,
      tune.early.stop.round = 50,
      annoying = T
    )
    
    xgb.params$tune.folds = list()
    date.quantiles = as.IDate(quantile(as.integer(train.meta$date_time), (1:2)/3), origin = as.IDate('1970-01-01'))
    xgb.params$tune.folds[[1]] = which(train.meta$date_time <= date.quantiles[1])
    xgb.params$tune.folds[[2]] = which(train.meta$date_time >  date.quantiles[1] & train.meta$date_time <= date.quantiles[2])
    xgb.params$tune.folds[[3]] = which(train.meta$date_time >  date.quantiles[2])
    
    if (do.autotune) {
      cat(date(), 'Tuning\n')
  
      #hp.grid = expand.grid(eta = c(0.02, 0.05, 0.1), max_depth = c(1, 4, 8))
      hp.grid = expand.grid(gamma = c(1, 10, 20), lambda = c(0.1, 1, 5))
      hp.grid$best.mlogloss = NA
      hp.grid$best.round = NA
      
      for (hp.i in 1:nrow(hp.grid)) {
        #cat(date(), 'Trying out: eta =', hp.grid$eta[hp.i], ', depth =', hp.grid$max_depth[hp.i], '\n')
        #xgb.params$eta       = hp.grid$eta      [hp.i]
        #xgb.params$max_depth = hp.grid$max_depth[hp.i]
        xgb.params$gamma      = hp.grid$gamma    [hp.i]
        xgb.params$lambda     = hp.grid$lambda   [hp.i]
        
        set.seed(1234)
        
        xgb.cv.res = xgb.cv(
          params = xgb.params, 
          nrounds = xgb.params$tune.nrounds,
          maximize = xgb.params$maximize,
          data = xtrain, 
          folds = xgb.params$tune.folds,
          prediction = F,
          early.stop.round = xgb.params$tune.early.stop.round, 
          verbose = F, 
          print.every.n = 10,
          nthread = 8
        )
        
        hp.grid$best.mlogloss[hp.i] =       min(xgb.cv.res$test.mlogloss.mean)
        hp.grid$best.round   [hp.i] = which.min(xgb.cv.res$test.mlogloss.mean)
        #cat(date(), 'Hyperconfig', hp.i, ':', hp.grid$best.mlogloss[hp.i], '@', hp.grid$best.round[hp.i], '\n')
      }
      
      hp.grid = hp.grid[order(hp.grid$best.mlogloss), ]
      #cat(date(), 'Best hyperconfig: eta =', hp.grid$eta[1], ', depth =', hp.grid$max_depth[1], '=>', hp.grid$best.mlogloss[1], '@', hp.grid$best.round[1], '\n')
      #xgb.params$eta       = hp.grid$eta       [1]
      #xgb.params$max_depth = hp.grid$max_depth [1]
      cat(date(), 'Best hyperconfig: gamma =', hp.grid$gamma[1], ', lambda =', hp.grid$lambda[1], '=>', hp.grid$best.mlogloss[1], '@', hp.grid$best.round[1], '\n')
      xgb.params$gamma     = hp.grid$gamma     [1]
      xgb.params$lambda    = hp.grid$lambda    [1]
      xgb.params$nrounds   = round(hp.grid$best.round[1] * 1.15)
    }

    #
    # Train
    #
    
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
      save_period       = 100
      #save_name         = path.expand(paste0(tmpdir, '/model.xgb'))
    )
    
    #if (0) {
    #  xgb.fit = xgb.load(path.expand(paste0(tmpdir, '/model.xgb')))
    #}
    
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
      top.subset.map5s[top.subset.i] = map5$value
      preds = t(matrix(preds, ncol = nrow(xvalid)))
      save(preds, file = paste0(tmpdir, '/xgb-', subset.on, '-', top.subset.i, '-preds-2014.RData'))
      
      preds = t(matrix(predict(xgb.fit, xtest), ncol = nrow(xtest)))
      save(preds, file = paste0(tmpdir, '/xgb-', subset.on, '-', top.subset.i, '-preds-2015v.RData'))
    } else {
      cat(date(), 'Predicting on test\n')
      preds = t(matrix(predict(xgb.fit, xtest), ncol = nrow(xtest)))
      save(preds, file = paste0(tmpdir, '/xgb-', subset.on, '-', top.subset.i, '-preds-2015.RData'))
    }
  }
  
  if (do.validation) {
    cat('\n', date(), 'All validation MAP5 results:\n\n')
    print(data.frame(ref = top.subset.ref.map5s, xgb = top.subset.map5s, benefit = top.subset.map5s - top.subset.ref.map5s))
  }
}

if (do.compose) {
  # Go over the top subsets and compare each XGB with the plug-in estimator on
  # the validation set, to select the best model. Then take the test predictions
  # for the selected model and produce one big happy matrix of predictions.

  cat(date(), 'Composing plugin and XGB with select best\n')
  
  #
  # Figure out which examples belong to each subset
  #
  
  load(paste0(tmpdir, '/pp-data.RData')) # => dat.model, dat.meta

  if (subset.on == 'market') {
    d.sub = dat.meta[, .N, by = hotel_market]
  } else if (subset.on == 'dest') {
    d.sub = dat.meta[, .N, by = srch_destination_id]
  } else {
    stop('WTF')
  }
  
  d.sub = d.sub[order(N, decreasing = T)]

  valid.idxs = (dat.meta$date_time >= first.valid.date) & (dat.meta$date_time < first.test.date)
  test.idxs  = (dat.meta$date_time >= first.test.date )
  valid.labels = dat.meta$hotel_cluster[valid.idxs]

  top.subset.valid.idxs = list()
  top.subset.test.idxs = list()
  for (top.subset.i in top.subset.is) {
    # FIXME this assumes splitting by dest id
    top.subset.valid.idxs[[top.subset.i]] = (dat.meta$srch_destination_id[valid.idxs] == d.sub$srch_destination_id[top.subset.i])
    top.subset.test.idxs [[top.subset.i]] = (dat.meta$srch_destination_id[test.idxs ] == d.sub$srch_destination_id[top.subset.i])
  }
  
  rm(dat.model, dat.meta, d.sub, valid.idxs, test.idxs)
  
  #
  # Model selection
  #

  load(paste0(tmpdir, '/desmar-preds-2014.RData')) # => preds
  plugin.valid.preds = preds
  rm(preds)
  load(paste0(tmpdir, '/desmar-preds-2015.RData')) # => preds
  plugin.test.preds = preds
  rm(preds)
  
  eval.map5.core = function(preds, labels) {
    preds = t(apply(preds, 1, order, decreasing = T))[, 1:5] - 1
    succ = (preds == labels)
    w = 1 / (1:5)
    return (mean(succ %*% w))
  }
  
  for (top.subset.i in top.subset.is) {
    cat(date(), 'Working on subset', top.subset.i, '\n')
    valid.idxs.i = top.subset.valid.idxs[[top.subset.i]]
    test.idxs.i  = top.subset.test.idxs [[top.subset.i]]
    plugin.valid.preds.i = plugin.valid.preds[valid.idxs.i, ]

    load(paste0(tmpdir, '/xgb-', subset.on, '-', top.subset.i, '-preds-2014.RData')) # => preds
    xgb.valid.preds.i = preds
    rm(preds)
    
    map5.plugin.i = eval.map5.core(plugin.valid.preds.i, valid.labels[valid.idxs.i])
    map5.xgb.i    = eval.map5.core(xgb.valid.preds.i   , valid.labels[valid.idxs.i])
    
    if (map5.xgb.i > map5.plugin.i) {
      plugin.valid.preds[valid.idxs.i, ] = xgb.valid.preds.i
      load(paste0(tmpdir, '/xgb-', subset.on, '-', top.subset.i, '-preds-2015.RData')) # => preds
      preds = as.data.table(preds)
      setnames(preds, names(plugin.test.preds))
      plugin.test.preds[test.idxs.i] = preds
    }
  }
  
  cat(date(), 'Combined MAP5 (optimistic due to selection):', eval.map5.core(plugin.valid.preds, valid.labels), '\n')
  preds = plugin.test.preds
  rm(plugin.test.preds)
  save(preds, file = paste0(tmpdir, '/desmar-xgb-preds-2015.RData'))
}

cat(date(), 'Done.\n')
