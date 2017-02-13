# Notes from the forums:
# - the data in the destination file is apparently log10 of probabilities that sum to 1
# - timestamps are in the timezone of the site_name
# - since it's the first time I'm working with the MAP@k metric in this setting: it simplifies to finding the top k best candidates
# - maybe use library(feather) for fast loading/storing data to disk and R interoperability [this requires R3.3 with new toolchain... need to wait for it to come out]
# - country/region/continent are (naturally) categorical
# - need to add to the testset: is_booking=1 (exact), cnt=1 (approximation)
# - timestamps for multiple clicks with no booking only refer to the first click
# - annoyingly, different id features that should match were not coded the same way, e.g., user_location_country_id and hotel_country_id (are there other examples?)
# - there is a critical leak in the data, in that orig_destination_distance can be used to identify hotels (and thus the cluster)
# - a given hotel can be associated with different clusters at different times (e.g., seasonal changes, or slow evolution)
# - search destinations are nodes on a tree of overlapping geographical locations (e.g. NY->NYC->Manhattan,JFK kind of). We don't know the tree (but maybe we can learn it from distances...)
# - markets on the other hand are non overlapping
# - the orig_destination_distance is in miles, and represents flight distance with stops (or something like that)
# - regions (and I think cities) are coded by their string name. Sometimes the same name (something like "center" or "london") will appear in multiple countries (so need to consider [country, region] and [country, region, city] as atomic factors)
# - the public/private split is time-based, meaning (I assume) that the split point is 2015-04-24

# Random notes to self:
# - hotel_markets are associated with a single continent, country (with <20 exceptions, maybe small/sparse countries, or market == 0)
# - on the other hand, there are many more srch_destination_ids than there are hotel_markets. Almost all destination_ids are asspcoated with a single market.
# - distances are available for is_mobile and not (so maybe it's often based on user address or ip)
# - posa_continent is redundant when we have site_name, it should've been given in a separate table (but it groups site names for us)
# - very few users have more than 10 booking events, and it grows exponentially, so that most users only have one booking event
# - still, I will probably want to add timeseries features like 'last cluster booked/searched' etc.
# - unsurprisingly, almost all users are always associated with the same country/region/city/posa
# - I can map hotel and user countries/continents (maybe cities and markets too) using distances where available

library (data.table)

#load(file = 'c:/TEMP/kaggle/expedia-hotel-recommendations/raw.RData') # => train, test
load(file = '~/Scratch/Kaggle/expedia-hotel-recommendations/raw.RData') # => train, test

# So it will be easier to rbind
test$hotel_cluster = NA
test$is_booking = 1

# Look at the marginal distribution of clusters
tbl = table(train$hotel_cluster)
plot(tbl)
# => there are differences, but they are not orders of magnitude, so the classification is obviously
# meant to achieve this.

# Are there differences between the distribution of clusters in the "public" and "private" parts of the year?
xtr = c(data.matrix(config$dtr$hotel_cluster[idx.pub]))
xte = c(data.matrix(config$dtr$hotel_cluster[!idx.pub]))
brks = sort(unique(c(xtr, xte)))
brks = c(brks[1] - 0.5, brks[-length(brks)] + diff(brks) / 2, tail(brks, 1) + 0.5)
tbl.train = hist(xtr, breaks = brks, plot = F)
tbl.test  = hist(xte, breaks = brks, plot = F)
plot(tbl.train, col = rgb(0, 0, 1, 1/4), freq = F, ylim = range(tbl.train$density, tbl.test$density, na.rm = T))
plot(tbl.test , col = rgb(1, 0, 0, 1/4), freq = F, add = T)
plot(tbl.train$density - tbl.test$density)
qqnorm(tbl.train$density - tbl.test$density)
qqline() # => seems like at least two clusters have significant differences. Not obvious how to exploit this.

# Do some/all of the testset users appear in the trainset?
mean(test$user_id %in% train$user_id) 
# => hmm... so the users in these data are such that they clicked / booked at least once in both
# train and test periods. The train/test split though is only based on clendar time, so different
# users wildly varying ratios of train to test number of events. Note that some users don't have
# any bookings in train, only clicks.

view.one.user = function(user.id, only.bookings = F) {
  if (only.bookings) {
    View(rbind(train[user_id == user.id & is_booking == T, .(date_time, hotel_market, hotel_cluster)], 
               test [user_id == user.id & is_booking == T, .(date_time, hotel_market, hotel_cluster)]))
  } else {
    View(rbind(train[user_id == user.id, .(date_time, is_booking, hotel_market, hotel_cluster)], 
               test [user_id == user.id, .(date_time, is_booking, hotel_market, hotel_cluster)]))
  }
}

# Compare marginal feature distributions in train and test (at least those easy enough to work 
# with) - are there any obvious differences?
# FIXME maybe it makes more sense to compair only bookings from train
feats = c('site_name', 'user_location_country', 'is_mobile', 'is_package', 'channel', 'srch_adults_cnt', 'srch_children_cnt', 'srch_rm_cnt', 'srch_destination_type_id', 'hotel_continent', 'hotel_country')
for (f in feats) {
  xtr = c(data.matrix(train[, f, with = F]))
  xte = c(data.matrix(test [, f, with = F]))
  brks = sort(unique(c(xtr, xte)))
  brks = c(brks[1] - 0.5, brks[-length(brks)] + diff(brks) / 2, tail(brks, 1) + 0.5)
  tbl.train = hist(xtr, breaks = brks, plot = F)
  tbl.test  = hist(xte, breaks = brks, plot = F)
  plot(tbl.train, col = rgb(0, 0, 1, 1/4), freq = F, ylim = range(tbl.train$density, tbl.test$density, na.rm = T), main = f)
  plot(tbl.test , col = rgb(1, 0, 0, 1/4), freq = F, add = T)
}
# => user countries change from train to test (not huge but significant)
# => there's a very small increase in mobile usage
# => many more packages in train than in test (difference is smaller when only considering bookings)
# => the channel distribution is totally different! channel 10 became the main one in test, but was negligible in train
# => there are more singleton searches in test
# => in particular searches with children are less common in test
# => I don't expect it to be very important, but destination types are different too (maybe due to website design changes?)
# => there is some visible changes in the hotel continent & country distribution (most prominantly in the top country [USA?])

# Are there hotel_clusters where people don't book when traveling with kids/partners?
plot(table(train$hotel_cluster, train$srch_children_cnt))
plot(table(train$hotel_cluster, train$srch_adults_cnt))
# => well, yes, but only as part of a proper model.. (i.e., these features will probably be useful, mostly srch_children_cnt)

# Look at response dependences with a few other low cardinality features
plot(table(train$hotel_cluster, train$srch_destination_type_id)) # => useful, in a model..
plot(table(train$hotel_cluster, train$is_package)) # => strong effect, but still need a model
plot(table(train$hotel_cluster, train$channel)) # => no effect?
plot(table(train$hotel_cluster, train$srch_rm_cnt)) # => no effect?
plot(table(train$hotel_cluster, train$hotel_continent)) # => useful, in a model..

# Is there a big impact on cluster for the number of user events?
train[, nr.user.events := .N, by = .(user_id, is_booking)]
hist(train$nr.user.events)
plot(table(train$hotel_cluster[train$is_booking == 1], Hmisc::cut2(train$nr.user.events[train$is_booking == 1], g = 3))) # => doesn't look useful (but might be useful in interaction with other features)

# Is there a big difference in distribution of clusters during xmas? (as an example of a big holiday)
train[, srch_ci.date := as.IDate(srch_ci)]
xtr = c(data.matrix(train[is_booking == 1 & (week(srch_ci.date) == 1 | week(srch_ci.date) > 49), hotel_cluster]))
xte = c(data.matrix(train[is_booking == 1 & (week(srch_ci.date) > 1 & week(srch_ci.date) <= 49), hotel_cluster]))
brks = sort(unique(c(xtr, xte)))
brks = c(brks[1] - 0.5, brks[-length(brks)] + diff(brks) / 2, tail(brks, 1) + 0.5)
tbl.train = hist(xtr, breaks = brks, plot = F)
tbl.test  = hist(xte, breaks = brks, plot = F)
plot(tbl.train, col = rgb(0, 0, 1, 1/4), freq = F, ylim = range(tbl.train$density, tbl.test$density, na.rm = T), main = f)
plot(tbl.test , col = rgb(1, 0, 0, 1/4), freq = F, add = T)
# => many of these differences are significant. This can probably be the basis for an is_business_hotel feature (put NAs when difference is insignificant)

#
# Try to map user and hotel countries
#

country.dists = rbind(train[, .(orig_destination_distance, user_location_country, hotel_country)], test[, .(orig_destination_distance, user_location_country, hotel_country)])
country.dists = train[, .(mean.dist = mean(orig_destination_distance, na.rm = T)), by = .(user_location_country, hotel_country)]
country.dists = country.dists[!is.na(mean.dist)]
country.dists = country.dists[order(user_location_country, mean.dist)]
closest.countries = country.dists[!duplicated(user_location_country)][, 1:2, with = F]
save(closest.countries, file = 'closest-countries.RData')

#
# Measure median distances of user cities and hotel markets
#

city.dists = rbind(train[!is.na(orig_destination_distance), .(orig_destination_distance, user_location_country, user_location_region, user_location_city, hotel_market)], 
                   test [!is.na(orig_destination_distance), .(orig_destination_distance, user_location_country, user_location_region, user_location_city, hotel_market)])

city.dists = train[, .(
  q0 = min(orig_destination_distance, na.rm = T),
  q1 = quantile(orig_destination_distance, 0.25, na.rm = T),
  q2 = median(orig_destination_distance, na.rm = T),
  q3 = quantile(orig_destination_distance, 0.75, na.rm = T),
  q4 = max(orig_destination_distance, na.rm = T)), 
  by = .(user_location_country, user_location_region, user_location_city, hotel_market)]

city.dists = city.dists[is.finite(q0)]

save(city.dists, file = 'city-dists.RData')

#
# Generate a simple cluster likelihood per destination
#

# FIXME: I can include target-revealed-by-distance-leak test examples in this computation

w1 = 4
dest.yenc = train[, .(w = .N + (w1 - 1) * sum(is_booking)), by = .(srch_destination_id, hotel_cluster)]
dest.yenc = dcast(dest.yenc, srch_destination_id ~ hotel_cluster, value.var = 'w', fill = 0)
setnames(dest.yenc, c('srch_destination_id', paste0('dest.yenc', 0:99)))
save(dest.yenc, file = 'dest-yenc.RData')

#
# Look at the destinations table
#

# The major question seems to be: can I cluster the destination ids? We've seen that used as-is 
# they are very useful.

dests = fread('c:/TEMP/kaggle/expedia/destinations.csv', header = T)
# These dimensions describe (log10) probabilities of the destination belonging to one of 150 
# Expedia-proprietary classes. The classes must be mutually exclusive, otherwise assigning 
# probabilities that sum to 1 makes no sense. The description says these were extracted from reviews
# of hotels in each destination.
dests = 10 ^ data.matrix(dests)[, -1]
colMeans(dests) # => so generally speaking the 150 classes have similar mass, i.e., destination are distributed roughly uniformly across the classes

# I'm not sure if it makes sense to use PCA, or any other dim reduct method. Maybe...
prcomp.res = prcomp(dests, center = T, scale. = T)
save(prcomp.res, file = 'pca.RData')

plot(prcomp.res, type = 'l', log = 'y', npcs = 80) # => take the top 9?
# => maybe not surprising, it seems like there is dependence between destinations. It seems like a
# good idea to include the top PCs in addition / instead of the destination id (which is a very
# high cardinality categorical) in the model.

plot(as.data.frame(prcomp.res$x[, 1:4]), pch = '.') # => I like (PC2, PC4) especially
# TODO: merge this info with the full data, and see how it relates to the target

library (Rtsne)
set.seed(1234)
tsne.res = Rtsne(dests, check_duplicates = F, pca = T, max_iter = 5000, perplexity = 30, theta = 0.5, dims = 2, verbose = T)
save(tsne.res, file = 'tsne.RData')

plot(tsne.res$Y[, 1], tsne.res$Y[, 2], pch = '.') # => this seems pretty useless
