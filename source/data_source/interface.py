class DataSourceInterface:
    def __init__(self):
        raise NotImplementedError("Must read constructor.");

    def read(self):
        """
        Read data entry.
        === OUTPUT ===
        a dictionary that includes:
            "time": the timestamp of the data, in s;
            "gyro": [x, y, z] the raw reading of the strapdown gyro, in rad/s;
            "accel": [x, y, z] the raw reading of the strapdown accelerometer, in m^2/s;
            "mag": [x, y, z] the raw reading of the strapdown magnetometer, in G(gauss);
        or None if eof.
        """
        raise NotImplementedError("Must implement read().");

    def eof(self):
        """
        Check to see if there is more data.
        === OUTPUT ===
        false if no data available.
        """

if (__name__ == "__main__"):
    pass
