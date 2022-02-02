import numpy as np
import pickle
import copy


class ILayer:
    def forward(self, i, t):
        self.output = i


class DLayer:
    def __init__(self, ni, nn, wl1=0, wl2=0, bl1=0, bl2=0):
        self.weights = 0.1 * np.random.randn(ni, nn)
        self.biases = np.zeros((1, nn))
        self.wl1 = wl1
        self.wl2 = wl2
        self.bl1 = bl1
        self.bl2 = bl2

    def forward(self, i, t):
        self.inputs = i
        self.output = np.dot(i, self.weights) + self.biases

    def backward(self, dv):
        self.dweights = np.dot(self.inputs.T, dv)
        self.dbiases = np.sum(dv, axis=0, keepdims=True)

        if self.wl1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.wl1 * dL1

        if self.wl2 > 0:
            self.dweights += 2 * self.wl2 * self.weights

        if self.bl1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bl1 * dL1

        if self.bl2 > 0:
            self.dbiases += 2 * self.bl2 * self.biases

        self.dinputs = np.dot(dv, self.weights.T)

    def get_params(self):
        return self.weights, self.biases

    def set_params(self, w, b):
        self.weights = w
        self.biases = b


class LayerD:
    def __init__(self, r):
        self.r = 1 - r

    def forward(self, i, t):
        self.inputs = i

        if not t:
            self.output = i.copy()
            return

        self.bm = np.random.binomial(1, self.r, size=i.shape) / self.r

        self.output = i * self.bm

    def backward(self, dv):
        self.dinputs = dv * self.bm


class Act_ReLU:
    def forward(self, i, t):
        self.inputs = i
        self.output = np.maximum(0, i)

    def backward(self, dv):
        self.dinputs = dv.copy()
        self.dinputs[self.inputs <= 0] = 0

    def predictions(self, o):
        return o


class Act_Softmax:
    def forward(self, i, t):
        self.inputs = i
        ev = np.exp(i - np.max(i, axis=1, keepdims=True))
        p = ev / np.sum(ev, axis=1, keepdims=True)

        self.output = p

    def backward(self, dv):
        self.dinputs = np.empty_like(dv)

        for i, (so, sd) in enumerate(zip(self.output, dv)):
            so = so.reshape(-1, 1)
            jm = np.diagflat(so) - np.dot(so, so.T)

            self.dinputs[i] = np.dot(jm, sd)

    def predictions(self, o):
        return np.argmax(o, axis=1)


class Act_Sigmoid:
    def forward(self, i, t):
        self.inputs = i
        self.output = 1 / (1 + np.exp(-i))

    def backward(self, dv):
        self.dinputs = dv * (1 - self.output) * self.output

    def predictions(self, o):
        return (o > 0.5) * 1


class Act_Linear:
    def forward(self, i, t):
        self.inputs = i
        self.output = i

    def backward(self, dv):
        self.dinputs = dv.copy()

    def predictions(self, o):
        return o


class Loss:
    def regularization_loss(self):
        rl = 0

        for layer in self.trainable_layers:
            if layer.wl1 > 0:
                rl += layer.wl1 * np.sum(np.abs(layer.weights))
            if layer.wl2 > 0:
                rl += layer.wl2 * np.sum(layer.weights * layer.weights)
            if layer.bl1 > 0:
                rl += layer.bl1 * np.sum(np.abs(layer.biases))
            if layer.bl2 > 0:
                rl += layer.bl2 * np.sum(layer.biases * layer.biases)

        return rl

    def remember_trainable_layers(self, tl):
        self.trainable_layers = tl

    def calculate(self, o, y, *, ir=False):
        sl = self.forward(o, y)
        dl = np.mean(sl)

        self.asum += np.sum(sl)
        self.ac += len(sl)

        if not ir:
            return dl

        return dl, self.regularization_loss()

    def calculate_accumulated(self, *, ir=False):
        dl = self.asum / self.ac

        if not ir:
            return dl

        return dl, self.regularization_loss()

    def new_pass(self):
        self.asum = 0
        self.ac = 0


class Loss_MSE(Loss):
    def forward(self, yp, yt):
        sl = np.mean((yt - yp) ** 2, axis=1)

        return sl

    def backward(self, dv, yt):
        s = len(dv)
        o = len(dv[0])

        self.dinputs = -2 * (yt - dv) / o
        self.dinputs /= s


class Loss_MAE(Loss):
    def forward(self, yp, yt):
        sl = np.mean(np.abs(yt - yp), axis=-1)

        return sl

    def backward(self, dv, yt):
        s = len(dv)
        o = len(dv[0])

        self.dinputs = np.sign(yt - dv) / o
        self.dinputs /= s


class Loss_LBC(Loss):
    def forward(self, yp, yt):
        ypc = np.clip(yp, 1e-7, 1 - 1e-7)

        sl = -(yt * np.log(ypc) + (1 - yt) * np.log(1 - ypc))
        sl = np.mean(sl, axis=-1)

        return sl

    def backward(self, dv, yt):
        s = len(dv)
        o = len(dv[0])

        cdv = np.clip(dv, 1e-7, 1 - 1e-7)

        self.dinputs = -(yt / cdv - (1 - yt) / (1 - cdv)) / o
        self.dinputs /= s


class Loss_LCC(Loss):
    def forward(self, yp, yt):
        s = len(yp)

        ypc = np.clip(yp, 1e-7, 1 - 1e-7)

        cc = None
        if len(yt.shape) == 1:
            cc = ypc[range(s), yt]
        elif len(yt.shape) == 2:
            cc = np.sum(ypc * yt, axis=1)

        nll = -np.log(cc)

        return nll

    def backward(self, dv, yt):
        s = len(dv)
        ls = len(dv[0])

        if len(yt.shape) == 1:
            yt = np.eye(ls)[yt]

        self.dinputs = -yt / dv
        self.dinputs /= s


class Loss_SLCC():
    def backward(self, dv, yt):
        s = len(dv)

        if len(yt.shape) == 2:
            yt = np.argmax(yt, axis=1)

        self.dinputs = dv.copy()
        self.dinputs[range(s), yt] -= 1
        self.dinputs /= s


class Opt_SGC:
    def __init__(self, lr=1., d=0., m=0.):
        self.lr = lr
        self.clr = lr
        self.d = d
        self.i = 0
        self.m = m

    def pre_update_params(self):
        if self.d:
            self.clr = self.lr * (1. / (1. + self.d * self.i))

    def update_params(self, la):
        if self.m:
            if not hasattr(la, "weight_momentums"):
                la.weight_momentums = np.zeros_like(la.weights)
                la.bias_momentums = np.zeros_like(la.biases)

            wu = self.m * la.weight_momentums - self.clr * la.dweights
            la.weight_momentums = wu
            bu = self.m * la.bias_momentums - self.clr * la.dbiases
            la.bias_momentums = bu
        else:
            wu = -self.clr * la.dweights
            bu = -self.clr * la.dbiases

        la.weights += wu
        la.biases += bu

    def post_update_params(self):
        self.i += 1


class Opt_AG:
    def __init__(self, lr=1., d=0., ep=1e-7):
        self.lr = lr
        self.clr = lr
        self.d = d
        self.i = 0
        self.ep = ep

    def pre_update_params(self):
        if self.d:
            self.clr = self.lr * (1. / (1. + self.d * self.i))

    def update_params(self, la):
        if not hasattr(la, "weight_cache"):
            la.weight_cache = np.zeros_like(la.weights)
            la.bias_cache = np.zeros_like(la.biases)

        la.weight_cache += la.dweights ** 2
        la.bias_cache += la.dbiases ** 2

        la.weights += -self.clr * la.dweights / (np.sqrt(la.weight_cache) + self.ep)
        la.biases += -self.clr * la.dbiases / (np.sqrt(la.bias_cache) + self.ep)

    def post_update_params(self):
        self.i += 1


class Opt_RMSP:
    def __init__(self, lr=0.001, d=0., ep=1e-7, rho=0.9):
        self.lr = lr
        self.clr = lr
        self.d = d
        self.i = 0
        self.ep = ep
        self.rho = rho

    def pre_update_params(self):
        if self.d:
            self.clr = self.lr * (1. / (1. + self.d * self.i))

    def update_params(self, la):
        if not hasattr(la, "weight_cache"):
            la.weight_cache = np.zeros_like(la.weights)
            la.bias_cache = np.zeros_like(la.biases)

        la.weight_cache = self.rho * la.weight_cache + (1 - self.rho) * la.dweights ** 2
        la.bias_cache = self.rho * la.bias_cache + (1 - self.rho) * la.dbiases ** 2

        la.weights += -self.clr * la.dweights / (np.sqrt(la.weight_cache) + self.ep)
        la.biases += -self.clr * la.dbiases / (np.sqrt(la.bias_cache) + self.ep)

    def post_update_params(self):
        self.i += 1


class Opt_Adam:
    def __init__(self, lr=0.001, d=0., ep=1e-7, b1=0.9, b2=0.999):
        self.lr = lr
        self.clr = lr
        self.d = d
        self.i = 0
        self.ep = ep
        self.b1 = b1
        self.b2 = b2

    def pre_update_params(self):
        if self.d:
            self.clr = self.lr * (1. / (1. + self.d * self.i))

    def update_params(self, la):
        if not hasattr(la, "weight_cache"):
            la.weight_momentums = np.zeros_like(la.weights)
            la.weight_cache = np.zeros_like(la.weights)
            la.bias_momentums = np.zeros_like(la.biases)
            la.bias_cache = np.zeros_like(la.biases)

        la.weight_momentums = self.b1 * la.weight_momentums + (1 - self.b1) * la.dweights
        la.bias_momentums = self.b1 * la.bias_momentums + (1 - self.b1) * la.dbiases

        weight_momentums_corrected = la.weight_momentums / (1 - self.b1 ** (self.i + 1))
        bias_momentums_corrected = la.bias_momentums / (1 - self.b1 ** (self.i + 1))

        la.weight_cache = self.b2 * la.weight_cache + (1 - self.b2) * la.dweights ** 2
        la.bias_cache = self.b2 * la.bias_cache + (1 - self.b2) * la.dbiases ** 2

        weight_cache_corrected = la.weight_cache / (1 - self.b2 ** (self.i + 1))
        bias_cache_corrected = la.bias_cache / (1 - self.b2 ** (self.i + 1))

        la.weights += -self.clr * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.ep)
        la.biases += -self.clr * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.ep)

    def post_update_params(self):
        self.i += 1


class A:
    def calculate(self, p, y):
        c = self.compare(p, y)

        a = np.mean(c)

        self.asum += np.sum(c)
        self.ac += len(c)

        return a

    def calculate_accumulated(self):
        a = self.asum / self.ac

        return a

    def new_pass(self):
        self.asum = 0
        self.ac = 0


class Acc_Regression(A):
    def __init__(self):
        self.p = None

    def init(self, y, reinit=False):
        if self.p is None or reinit:
            self.p = np.std(y) / 250

    def compare(self, p, y):
        return np.absolute(p - y) < self.p


class Acc_Categorical(A):
    def __init__(self, *, binary=False):
        self.b = binary

    def init(self, y):
        pass

    def compare(self, p, y):
        if not self.b and len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        return p == y


class Model:
    def __init__(self):
        self.layers = []
        self.softmax_classifier_output = None

    def add(self, layer):
        self.layers.append(layer)

    def set(self, *, loss=None, optimizer=None, accuracy=None):
        if loss is not None:
            self.loss = loss

        if optimizer is not None:
            self.optimizer = optimizer

        if accuracy is not None:
            self.accuracy = accuracy

    def finalize(self):
        self.input_layer = ILayer()

        layer_count = len(self.layers)

        self.trainable_layers = []

        for i in range(layer_count):
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i + 1]

            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.layers[i + 1]

            else:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]

            if hasattr(self.layers[i], "weights"):
                self.trainable_layers.append(self.layers[i])

        if self.loss is not None:
            self.loss.remember_trainable_layers(self.trainable_layers)

        if isinstance(self.layers[len(self.layers) - 1], Act_Softmax) and isinstance(self.loss, Loss_LCC):
            self.softmax_classifier_output = Loss_SLCC()

    def train(self, X, y, *, epochs=1, batch_size=None, print_every=1, validation_data=None):
        self.accuracy.init(y)

        train_steps = 1

        if validation_data is not None:
            validation_steps = 1

            X_val, y_val = validation_data

        if batch_size is not None:
            train_steps = len(X) // batch_size

            if train_steps * batch_size < len(X):
                train_steps += 1

            if validation_data is not None:
                validation_steps = len(X_val) // batch_size

                if validation_steps * batch_size < len(X_val):
                    validation_steps += 1

        for epoch in range(1, epochs + 1):
            print(f'epoch: {epoch}')

            self.loss.new_pass()
            self.accuracy.new_pass()

            for step in range(train_steps):
                if batch_size is None:
                    batch_X = X
                    batch_y = y

                else:
                    batch_X = X[step * batch_size:(step + 1) * batch_size]
                    batch_y = y[step * batch_size:(step + 1) * batch_size]

                output = self.forward(batch_X, training=True)

                data_loss, regularization_loss = self.loss.calculate(output, batch_y, ir=True)
                loss = data_loss + regularization_loss

                predictions = self.output_layer_activation.predictions(output)
                accuracy = self.accuracy.calculate(predictions, batch_y)

                self.backward(output, batch_y)

                self.optimizer.pre_update_params()
                for layer in self.trainable_layers:
                    self.optimizer.update_params(layer)
                self.optimizer.post_update_params()

                if not step % print_every or step == train_steps - 1:
                    print(f'step: {step}, ' +
                          f'acc: {accuracy:.3f}, ' +
                          f'loss: {loss:.3f} (' +
                          f'data_loss: {data_loss:.3f}, ' +
                          f'reg_loss: {regularization_loss:.3f}), ' +
                          f'lr: {self.optimizer.clr}')

            epoch_data_loss, epoch_regularization_loss = self.loss.calculate_accumulated(ir=True)
            epoch_loss = epoch_data_loss + epoch_regularization_loss
            epoch_accuracy = self.accuracy.calculate_accumulated()

            print(f'training, ' +
                  f'acc: {epoch_accuracy:.3f}, ' +
                  f'loss: {epoch_loss:.3f} (' +
                  f'data_loss: {epoch_data_loss:.3f}, ' +
                  f'reg_loss: {epoch_regularization_loss:.3f}), ' +
                  f'lr: {self.optimizer.clr}')

            if validation_data is not None:
                self.evaluate(*validation_data, batch_size=batch_size)

    def forward(self, X, training):
        self.input_layer.forward(X, training)
        for layer in self.layers:
            layer.forward(layer.prev.output, training)

        return layer.output

    def backward(self, output, y):
        if self.softmax_classifier_output is not None:
            self.softmax_classifier_output.backward(output, y)

            self.layers[-1].dinputs = self.softmax_classifier_output.dinputs

            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)

            return

        self.loss.backward(output, y)

        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)

    def evaluate(self, X_val, y_val, *, batch_size=None):
        validation_steps = 1

        if batch_size is not None:
            validation_steps = len(X_val) // batch_size

            if validation_steps * batch_size < len(X_val):
                validation_steps += 1

        self.loss.new_pass()
        self.accuracy.new_pass()

        for step in range(validation_steps):
            if batch_size is None:
                batch_X = X_val
                batch_y = y_val
            else:
                batch_X = X_val[step * batch_size:(step + 1) * batch_size]
                batch_y = y_val[step * batch_size:(step + 1) * batch_size]

            output = self.forward(batch_X, training=False)

            self.loss.calculate(output, batch_y)

            predictions = self.output_layer_activation.predictions(output)

            self.accuracy.calculate(predictions, batch_y)

        validation_loss = self.loss.calculate_accumulated()
        validation_accuracy = self.accuracy.calculate_accumulated()

        print(f'validation, ' +
              f'acc: {validation_accuracy:.3f}, ' +
              f'loss: {validation_loss:.3f}')

    def get_params(self):
        params = []

        for layer in self.trainable_layers:
            params.append(layer.get_params())

        return params

    def set_params(self, params):
        for ps, layer in zip(params, self.trainable_layers):
            layer.set_params(*ps)

    def save_params(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.get_params(), f)

    def load_params(self, path):
        with open(path, "rb") as f:
            self.set_params(pickle.load(f))

    def save(self, path):
        model = copy.deepcopy(self)

        model.loss.new_pass()
        model.accuracy.new_pass()

        model.input_layer.__dict__.pop("output", None)
        model.loss.__dict__.pop("dinputs", None)

        for layer in model.layers:
            for property in ["inputs", "output", "dinputs", "dweights", "dbiases"]:
                layer.__dict__.pop(property, None)

        with open(path, "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            model = pickle.load(f)

        return model

    def predict(self, X, *, batch_size=None):
        prediction_steps = 1

        if batch_size is not None:
            prediction_steps = len(X) // batch_size

            if prediction_steps * batch_size < len(X):
                prediction_steps += 1

        output = []

        for step in range(prediction_steps):
            if batch_size is None:
                batch_X = X
            else:
                batch_X = X[step * batch_size:(step + 1) * batch_size]

        batch_output = self.forward(batch_X, training=False)

        output.append(batch_output)

        return np.vstack(output)
