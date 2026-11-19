parameters {
  simplex[3] theta;
  array[2] vector<lower=0>[2] tau;
  ordered[2] c;
}
model {
  for (k in 1:3) target += k * log(theta[k]);
  for (i in 1:2) tau[i] ~ exponential(1);
  c ~ normal(0, 3);
}
