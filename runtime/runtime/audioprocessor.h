#ifndef AUDIOPROCESSOR_H
#define AUDIOPROCESSOR_H

#include <vector>
#include <exception>
#include <complex>
#include <valarray>

class MatrixMath{
private:
    /**
     * @brief Perform fast fourier transformation on given complex valarray.
     * @param complexFrame Samples of real signal and also a result of FFT after function call.
     */
    static void fft(std::valarray<std::complex<long double>> & complexFrame);

public:
    typedef std::vector<long double> vec;
    typedef std::vector<std::vector<long double>> vec2d;

    /**
     * @brief Perform transpose operation on given matrix.
     * @param v Matrix to transpose and transposed matrix after function call.
     */
    static void transposeMatrix(vec2d & v);

    /**
     * @brief Perform dot product on two matrices.
     * @param first First matrix in dot product and also result of dot product after function call.
     * @param second Second matrix in dot product operation.
     */
    static void dotMatrix(vec2d & first, const vec2d & second);

    /**
     * @brief Subtract given vector from every row in matrix.
     * @param first Matrix to subtract from and also result of subtraction after function call.
     * @param second Vector of values to subtract.
     */
    static void subtractMatrixByRows(vec2d & first, const vec & second);

    /**
     * @brief Remove columns from start and from end.
     * @param v Matrix to remove columns from and result of operation.
     * @param numFromStart Number of columns from start.
     * @param numToEnd Number of columns from end.
     */
    static void eraseColumnsMatrix(vec2d & v, unsigned int numFromStart = 0, unsigned int numToEnd = 0);

    /**
     * @brief Replace zeros in matrix to closest non zero value epsilon.
     * @param v Matrix to stabilize and also stabilized matrix after function call.
     */
    static void stabilizeMatrix(vec2d & v);

    /**
     * @brief Rescale matrix to be between minVal and maxVal.
     * @param v Matrix to rescale and result of operation.
     * @param minVal Lowest value in matrix.
     * @param maxVal Highest value in matrix.
     */
    static void rescaleMatrix(vec2d & v, long double minVal, long double maxVal);

    /**
     * @brief Compute mean of every column in matrix.
     * @param v Matrix to compute means.
     * @return Vector of mean values of matrix v.
     */
    static vec meansMatrixByColumns(const vec2d & v);

    /**
     * @brief Get minimum values from each column in matrix.
     * @param v Matrix to get min values.
     * @return Vector of min values from each column in matrix.
     */
    static vec minMatrixByColumns(const vec2d & v);

    /**
     * @brief Get maximum values from each column in matrix.
     * @param v Matrix to get max values.
     * @return Vector of max values from each column in matrix.
     */
    static vec maxMatrixByColumns(const vec2d & v);

    /**
     * @brief Normalize each column in matrix using x' = (x - mean(x)) / (maxx - minx).
     * @param v Matrix to normalize and result of operation.
     */
    static void normalizeMatrixByColumns(vec2d & v);

    /**
     * @brief Find smallest value in matrix.
     * @param Vector to search.
     * @return Smallest value in matrix.
     */
    static long double minMatrix(const MatrixMath::vec2d & v);

    /**
     * @brief Find biggest value in matrix.
     * @param Vector to search.
     * @return Biggest value in matrix.
     */
    static long double maxMatrix(const MatrixMath::vec2d & v);

    /**
     * @brief Perform fast fourier transformation on given vector.
     * @param frame Samples of real signal and also a result of FFT after function call.
     */
    static void fftVector(MatrixMath::vec & frame, unsigned int NFFT);

    /**
     * @brief Perform fast fourier transformation for every row in given matrix.
     * @param frames Matrix of rows and also result of FFT after function call.
     */
    static void fftMatrix(MatrixMath::vec2d & frames, unsigned int NFFT);

    /**
     * @brief Compute discrete cosine transform on given vector.
     * @param v Vector to compute DCT and result of operation after function call.
     */
    static void dctVector(MatrixMath::vec & v);

    /**
     * @brief Compute discrete cosine transform on every row of given matrix.
     * @param Matrix to compute DCT and result of operation after function call.
     */
    static void dctMatrix(MatrixMath::vec2d & frames);

    /**
     * @brief Create vector of equally spaced values.
     * @param low First element in vector.
     * @param high Last element in vector.
     * @param numPoints Number of elements in vector.
     * @return Vector of equally spaced values.
     */
    static vec linspace(long double low, long double high, unsigned int numPoints);

};

class AudioProcessorException: public std::exception
{
private:
  const char* w;
public:
  AudioProcessorException(const char* what):w(what){}
  virtual const char* what() const
  {
    return w;
  }
};

class AudioProcessor
{
public:
    typedef std::vector<unsigned int> byteVec;
    struct config{
        unsigned int bytesPerSample = 0;
        unsigned int numberOfChannels = 0;
        unsigned int sampleRate = 0;
        long double emphasisCoeff = 0;
        unsigned int framingSize = 0;
        unsigned int framingStride = 0;
        unsigned int NFFT = 0;
        unsigned int numberOfFilterBanks = 0;
        bool MFCC = false;
        unsigned int firstMFCC = 0;
        unsigned int lastMFCC = 0;
        bool sinLift = false;
        unsigned int cepLifter = 0;
        bool normalize = false;
        bool rescale = false;
        long double rescaleMin = 0;
        long double rescaleMax = 0;
    };

private:
    config conf;

    /**
     * @brief Validate configuration struct.
     * @return True if struct contains valid configuration.
     */
    bool validateConfig() const;

    /**
     * @brief Convert frequency value to Mel scale.
     * @param sample Value to convert.
     * @return Value on Mel scale.
     */
    static long double hzToMel(long double sample){return 2595*log10l(1 + sample/700);}

    /**
     * @brief Convert vector of frequency values to Mel scale.
     * @param v Vector to convert and also converted values after function call.
     */
    static void hzToMel(MatrixMath::vec & v){for(unsigned int i = 0; i < v.size(); i++){v[i] = hzToMel(v[i]);}}

    /**
     * @brief Convert Mel value to frequency.
     * @param sample Value to convert.
     * @return  Value on frequency scale.
     */
    static long double melToHz(long double sample){return 700*(pow(10, sample/2595) - 1);}

    /**
     * @brief Convert vector of Mel values to frequency values.
     * @param v Vector to convert and also converted values after function call.
     */
    static void melToHz(MatrixMath::vec & v){for(unsigned int i = 0; i < v.size(); i++){v[i] = melToHz(v[i]);}}

    /**
     * @brief Convert audio/pcm bytes into samples.
     * @param buffer Buffer of bytes to convert.
     * @return Vector of samples.
     */
    MatrixMath::vec bytesToSamples(const byteVec & buffer) const;

    /**
     * @brief Convert stereo (or multi channel) vector of samples into mono signal using mean of channel amplitudes.
     * @param sampleData Signal to convert and also result of operation after function call.
     */
    void channelsToMono(MatrixMath::vec & sampleData) const;

    /**
     * @brief Convert magnitudes of frequency domain signal into power spectrum.
     * @param magnitudes Magnitudes to convert and also power spectrum after function call.
     */
    static void magnitudeToPower(MatrixMath::vec2d & magnitudes);

    /**
     * @brief Frame given signal into frames of specified in config length and stride.
     * @param sampleData Samples to frame.
     * @return Matrix where each row contains single frame.
     */
    MatrixMath::vec2d frameSamples(MatrixMath::vec & sampleData) const;

    /**
     * @brief Apply pre emphasis filter to given signal.
     * @param sampleData Input signal and also filtered signal after function call.
     */
    void preEmphasis(MatrixMath::vec & sampleData) const;

    /**
     * @brief Apply Hamming window function to given matrix of frames.
     * @param frames Frames to apply Hamming window into and also modified frames after function call.
     */
    void hammingWindow(MatrixMath::vec2d & frames) const;

    /**
     * @brief Apply triangular filters to given vector.
     * @param v Vector to apply triangular filters to and result of operation after function call.
     */
    void filterBanks(MatrixMath::vec2d & v) const;

    /**
     * @brief Apply sinusoidal liftering to given matrix.
     * @param Matrix to apply liftering to and result of operation after function call
     */
    void sinLiftMatrix(MatrixMath::vec2d & v) const;

public:
    /**
     * @brief Class constructor.
     * @param
     */
    AudioProcessor(config c = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}){setConfig(c);}

    /**
     * @brief Get config of audio processor.
     * @return Config of audio processor.
     */
    config getConfig() const {return conf;}

    /**
     * @brief Set new config of audio processor.
     * @param c Config to set.
     */
    void setConfig(config c){conf = c;}

    /**
     * @brief Convert given audio/pcm buffer into using either MSFB or MFCC matrix.
     * @param buffer Buffer to process.
     * @param mfcc Set to true if compute MFCC.
     * @return Spectogram.
     */
    MatrixMath::vec2d processBuffer(const byteVec & buffer) const;
};

#endif // AUDIOPROCESSOR_H
