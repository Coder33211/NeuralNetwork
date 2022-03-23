function sum(a) {
  let v = 0;

  for (let i = 0; i < a.length; i++) {
    v += a[i];
  }

  return v;
}

function mean(a) {
  let v = [];
  for (let r = 0; r < a.length; r++) {
    for (let c = 0; c < a[0].length; c++) {
      v.push(a[r][c]);
    }
  }
  return sum(v) / v.length;
}

function transpose(a) {
  let m = [];

  for (let c = 0; c < a[0].length; c++) {
    m[c] = [];
    for (let r = 0; r < a.length; r++) {
      m[c][r] = a[r][c];
    }
  }

  return m;
}

function dot(a, b) {
  let m = [];

  let t = transpose(b);

  for (let ar = 0; ar < a.length; ar++) {
    let r = [];
    for (let br = 0; br < t.length; br++) {
      let s = [];
      for (let c = 0; c < a[0].length; c++) {
        s.push(a[ar][c] * t[br][c]);
      }
      let v = 0;
      for (let i = 0; i < s.length; i++) {
        v += s[i];
      }
      r.push(v);
    }
    m.push(r);
  }

  return m;
}

function op(a, b, w) {
  if (w == "a") {
    return a + b;
  } else if (w == "s") {
    return a - b;
  } else if (w == "m") {
    return a * b;
  } else if (w == "d") {
    return a / b;
  } else if (w == "p") {
    return a ** b;
  } else if (w == "max") {
    return Math.max(a, b);
  } else if (w == "min") {
    return Math.min(a, b);
  }
}

function math(a, b, w) {
  let m = [];

  if (typeof a == "object" && typeof b == "object") {
    let ash = [a.length, a[0].length];
    let bsh = [b.length, b[0].length];

    if (ash[0] == bsh[0] && ash[1] == bsh[1]) {
      for (let r = 0; r < a.length; r++) {
        m[r] = [];
        for (let c = 0; c < a[0].length; c++) {
          m[r][c] = op(a[r][c], b[r][c], w);
        }
      }
    } else if (ash[0] > bsh[0] && ash[1] == bsh[1]) {
      for (let r = 0; r < a.length; r++) {
        m[r] = [];
        for (let c = 0; c < a[0].length; c++) {
          m[r][c] = op(a[r][c], b[0][c], w);
        }
      }
    } else if (ash[0] == bsh[0] && bsh[1] == 1) {
      for (let r = 0; r < a.length; r++) {
        m[r] = [];
        for (let c = 0; c < a[0].length; c++) {
          m[r][c] = op(a[r][c], b[r][0], w);
        }
      }
    } else if (bsh[0] == 1 && bsh[1] == 1) {
      for (let r = 0; r < a.length; r++) {
        m[r] = [];
        for (let c = 0; c < a[0].length; c++) {
          m[r][c] = op(a[r][c], b[0][0], w);
        }
      }
    }
  } else if (typeof a != "object" && typeof b == "object") {
    for (let r = 0; r < b.length; r++) {
      m[r] = [];
      for (let c = 0; c < b[0].length; c++) {
        m[r][c] = op(a, b[r][c], w);
      }
    }
  } else if (typeof b != "object" && typeof a == "object") {
    for (let r = 0; r < a.length; r++) {
      m[r] = [];
      for (let c = 0; c < a[0].length; c++) {
        m[r][c] = op(a[r][c], b, w);
      }
    }
  }

  return m;
}

function tanh(x, d) {
  let p = math(Math.E, x, "p");
  let n = math(Math.E, math(-1, x, "m"), "p");
  let t = math(p, n, "s");
  let b = math(p, n, "a");
  let f = math(t, b, "d");

  let df = math(1, math(f, 2, "p"), "s");

  if (d) {
    return df;
  } else {
    return f;
  }
}

function mse(yt, yp, d) {
  let f = mean(math(math(yp, yt, "s"), 2, "p"));

  let df = math(2, math(yp, yt, "s"), "m");

  if (typeof yt == "object") {
    df = math(df, yt[0].length, "d");
  }

  if (d) {
    return df;
  } else {
    return f;
  }
}

function createValues(r, c) {
  let m = [];

  for (let ir = 0; ir < r; ir++) {
    m[ir] = [];
    for (let ic = 0; ic < c; ic++) {
      // m[ir][ic] = Math.random() - 0.5;
      m[ir][ic] = Math.random() - 0.5;
    }
  }

  return m;
}

class Layer {
  constructor(ni, nn) {
    this.weights = createValues(ni, nn);
    this.biases = createValues(1, nn);
  }

  forward(i) {
    this.input = i;
    this.output = math(dot(i, this.weights), this.biases, "a");

    return this.output;
  }

  backward(oe, lr) {
    let ie = dot(oe, transpose(this.weights));
    let we = dot(transpose(this.input), oe);

    let uws = math(lr, we, "m");
    let ubs = math(lr, oe, "m");

    this.weights = math(this.weights, uws, "s");
    this.biases = math(this.biases, ubs, "s");

    return ie;
  }
}

class Activation {
  constructor(a) {
    this.activation = a;
  }

  forward(i) {
    this.input = i;
    this.output = this.activation(i, false);

    return this.output;
  }

  backward(oe, lr) {
    return math(this.activation(this.input, true), oe, "m");
  }
}

class Model {
  constructor() {
    this.layers = [];
  }

  add(l) {
    this.layers.push(l);
  }

  set(l) {
    this.loss = l;
  }

  train(X, y, es, lr) {
    let s = X.length;

    for (let i = 0; i < es; i++) {
      let me = 0;
      let a = [];
      for (let j = 0; j < s; j++) {
        let o = [X[j]];

        for (let l = 0; l < this.layers.length; l++) {
          o = this.layers[l].forward(o);
        }

        let err;
        let errd;

        if (typeof y[j] == "object") {
          err = this.loss([y[j]], o, false);
          errd = this.loss([y[j]], o, true);
        } else {
          err = this.loss(y[j], o, false);
          errd = this.loss(y[j], o, true);
        }

        me += err;

        if (typeof y[j] == "object") {
          if (
            y[j].indexOf(Math.max(...y[j])) == o[0].indexOf(Math.max(...o[0]))
          ) {
            a.push(1);
          } else {
            a.push(0);
          }
        } else {
          if (Math.round(y[j]) == Math.round(o[0][0])) {
            a.push(1);
          } else {
            a.push(0);
          }
        }

        for (let l = this.layers.length - 1; l >= 0; l--) {
          errd = this.layers[l].backward(errd, lr);
        }
      }

      me /= s;
      a = sum(a) / a.length;

      console.log("Epoch:", i + 1);
      console.log("Error:", me);
      console.log("Accuracy:", a);
      console.log("Accuracy Percent: " + a * 100 + "%");
    }
  }

  predict(X) {
    let s = X.length;
    let r = [];

    for (let i = 0; i < s; i++) {
      let o = [X[i]];

      for (let l = 0; l < this.layers.length; l++) {
        o = this.layers[l].forward(o);
      }

      r.push(o);
    }

    return r;
  }
}
