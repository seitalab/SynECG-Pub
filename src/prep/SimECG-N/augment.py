import numpy as np
from scipy import signal as scipy_signal

class Augment:

    def __init__(self, freq):

        self.freq = freq

        self.scale_max = 4
        self.scale_min = 0.25

        self.sine_m_max = 1
        self.sine_m_min = 0
        self.sine_f_max = 0.02
        self.sine_f_min = 0.001

        self.sq_m_max = 1
        self.sq_m_min = 0
        self.sq_f_max = 0.1
        self.sq_f_min = 0.001

        self.wn_m_max = 0.05
        self.wn_m_min = 0.

        self.wnp_m_max = 0.25
        self.wnp_m_min = 0
        self.wnp_w_max = 0.2
        self.wnp_w_min = 0

    def _rand_val(self, max_val, min_val):
        return np.random.rand() * (max_val - min_val) + min_val

    def random_scale(self, X):
        """
        Randomly scale given batch.

        rand (0 - 1) * (max_val - min_val) -> 0 ~ max_val - min_val
        -> + min_val -> min_val ~ max_val

        Args:
            X (np.ndarray): Array of shape (sequence_length,)
        Returns:
            scaledX (np.ndarray): Batch of shape (num_sample, num_channel, sequence_length)
        """
        scaler = self._rand_val(self.scale_max, self.scale_min)
        return scaler * X

    def sine_noise(self, X):
        """
        Args:
            X: [batchsize, num_channel, sequence_length]
        Returns:
            X_sine:
        """
        seqlen = len(X)
        duration = seqlen / self.freq
        f_val = self._rand_val(self.sine_f_max, self.sine_f_min)
        m_val = self._rand_val(self.sine_m_max, self.sine_m_min)
        steps = np.linspace(0, 2 * np.pi * duration * f_val, seqlen)

        X_sine = X.copy()
        X_sine += m_val * np.sin(steps)
        return X_sine

    def square_noise(self, X):
        """
        Args:
            X: [batchsize, num_channel, sequence_length]
            M (float): Value for amplitude (value between 0 - 1).
            F (float): Value for frequency
        Returns:
            X_square:
        """
        seqlen = len(X)
        duration = seqlen / self.freq
        f_val = self._rand_val(self.sq_f_max, self.sq_f_min)
        m_val = self._rand_val(self.sq_m_max, self.sq_m_min)
        steps = np.linspace(0, 2 * np.pi * duration * f_val, seqlen)

        X_square = X.copy()
        X_square += m_val * scipy_signal.square(steps)
    
        return X_square

    def white_noise(self, X):
        """
        Args:
            X (np.ndarray): Batch of shape (num_sample, num_channel, sequence_length)
            M (float): Amplitude of white noise (Value between 0 - 1.)
        Returns:
            X_wn (np.ndarray): Batch of shape (num_sample, num_channel, sequence_length)
        """
        X_wn = X.copy()

        amp = self._rand_val(self.wn_m_max, self.wn_m_min)
        # gaussian_noise centered to 0
        white_noise = np.random.randn(X.shape[0])
        X_wn += white_noise * amp
        return X_wn

    def sine_noise_partial(self, X):
        """
        Args:
            X: [batchsize, num_channel, sequence_length]
            M: value between 0 - 1.
            F (float):
        Returns:
            X_sine_p:
        """
        X_sine_p = X.copy()
        seqlen = X.shape[0]

        duration = seqlen / self.freq
        f_val = self._rand_val(self.sine_f_max, self.sine_f_min)
        w_ratio = self._rand_val(self.sine_m_max, self.sine_m_min)

        steps = np.linspace(0, 2 * np.pi * duration * f_val, seqlen)
        sine_curve = np.sin(steps)

        width = int(seqlen * w_ratio)
        start = int(np.random.rand() * (seqlen - width))
        for w in range(width):
            X_sine_p[start+w] += sine_curve[w]
        return X_sine_p

    def square_noise_partial(self, X):
        """
        Args:
            X: [batchsize, num_channel, sequence_length]
        Returns:
            X_square_p:
        """
        X_square_p = X.copy()
        seqlen = X.shape[0]

        duration = seqlen / self.freq
        f_val = self._rand_val(self.sq_f_max, self.sq_f_min)
        w_ratio = self._rand_val(self.sq_m_max, self.sq_m_min)

        steps = np.linspace(0, 2 * np.pi * duration * f_val, seqlen)
        square_pulse = scipy_signal.square(steps)

        width = int(seqlen * w_ratio)
        start = int(np.random.rand() * (seqlen - width))
        for w in range(width):
            X_square_p[start+w] += square_pulse[w]
        return X_square_p

    def white_noise_partial(self, X):
        """
        Args:
            X: [batchsize, num_channel, sequence_length]
            M: Magnitude of partial noise, corresponding to width of sample.
        Returns:
            X_wnp:
        """
        X_wnp = X.copy()
        seqlen = X.shape[0]

        w_ratio = self._rand_val(self.wnp_w_max, self.wnp_w_min)
        m_val = self._rand_val(self.wnp_m_max, self.wnp_m_min)
        width = int(seqlen * w_ratio)
        start = int(np.random.rand() * (seqlen - width))
        white_noise = np.random.randn(width)

        for w in range(width):
            X_wnp[start+w] += white_noise[w]
        return X_wnp

    def rand_augment(self, X):
        """

        """
        func = np.random.choice(7)

        if func == 0:
            augX = self.random_scale(X)
        elif func == 1:
            augX = self.sine_noise(X)
        elif func == 2:
            augX = self.square_noise(X)
        elif func == 3:
            augX = self.white_noise(X)
        elif func == 4:
            augX = self.sine_noise_partial(X)
        elif func == 5:
            augX = self.square_noise_partial(X)
        elif func == 6:
            augX = self.white_noise_partial(X)
        return augX