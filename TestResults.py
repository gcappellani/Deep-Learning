######### epochs = 5, eye_state=True ##########

### No momentum
# alfa = .1 => .8 | .776 | .816 | .736 | .776  AVG = .781

### Simple momentum VS no momentum
# alfa = .1, beta = .1, gamma = 1 => .8 | .792 | .816 | .824 | .744  AVG = .795 (+0.014)
# alfa = .1, beta = .3, gamma = 1 => .568
# alfa = .1, beta = .8, gamma = 1 => .28

### EMWA momentum VS simple momentum
# alfa = .1, beta = .1, gamma = .8 => .84 | .728 | .824 | .696  AVG = .772 (-0.036)
# alfa = .1, beta = .1, gamma = .5 => .864 | .8 | .84 | .728 | .776  AVG = .801 (-0.007)

# alfa = .1, beta = .1, gamma = .4 => .856 | .848 | .856 | .736 | .792  AVG = .817 (+0.022)

# alfa = .1, beta = .1, gamma = .333 => .84 | .808 | .816 | .792  AVG = .814 (+0.006)
# alfa = .1, beta = .1, gamma = .3 => .808 | .768 | .848 | .848  AVG = .818 (+0.010)


######### epochs = 20, user_id=True ##########
# alfa = .1, beta = .1, gamma = .4 => .992


