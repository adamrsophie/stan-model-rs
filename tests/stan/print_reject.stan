data {
  real y;
}
parameters {
  real mu;
}
model {
  print("mu = ", mu);
  if (mu > 10) {
    reject("mu too large: ", mu);
  }
  y ~ normal(mu, 1);
}
