#include "audioprocessor.h"

#include <algorithm>

#include <iostream>
using namespace std;

void MatrixMath::fft(std::valarray<std::complex<long double>> & complexFrame) {
    // DFT
    const size_t N = complexFrame.size();
    if (N <= 1) return;

    // divide
    std::valarray<std::complex<long double>> even = complexFrame[std::slice(0, N/2, 2)];
    std::valarray<std::complex<long double>>  odd = complexFrame[std::slice(1, N/2, 2)];

    // conquer
    fft(even);
    fft(odd);

    // combine
    for (size_t k = 0; k < N/2; ++k)
    {
        std::complex<long double> t = std::polar(1.0L, -2 * 3.14159265358979323846264338328L * k / N) * odd[k];
        complexFrame[k    ] = even[k] + t;
        complexFrame[k+N/2] = even[k] - t;
    }
}

void MatrixMath::transposeMatrix(vec2d & v){
    vec2d res(v[0].size(), vec(v.size(),0));

    for(unsigned int i = 0; i < res.size(); i++){
        for(unsigned int j = 0; j < res[i].size(); j++){
            res[i][j] = v[j][i];
        }
    }

    v = std::move(res);
}

void MatrixMath::dotMatrix(vec2d & first, const vec2d & second) {
    vec2d res(first.size(), vec(second[0].size(), 0));

    // for each row in first array
    for(unsigned int i = 0; i < first.size(); i++){
        // for each column in second array
        for(unsigned int j = 0; j < second[0].size(); j++){
            long double sum = 0;
            // sum each element
            for(unsigned int k = 0; k < first[i].size(); k++){
                 sum += first[i][k] * second[k][j];
            }
            res[i][j] = sum;
        }
    }
    first.clear();
    first = std::move(res);
}

void MatrixMath::subtractMatrixByRows(vec2d & first, const vec & second){
    for(unsigned int i = 0; i < first.size(); i++){
        for(unsigned int j = 0; j < first[i].size(); j++){
            first[i][j] -= second[j];
        }
    }
}

void MatrixMath::eraseColumnsMatrix(vec2d & v, unsigned int numFromStart, unsigned int numToEnd){
    for(unsigned int i = 0; i < v.size(); i++){
        v[i].erase(v[i].begin(), v[i].begin() + numFromStart);
        v[i].erase(v[i].end() - numToEnd, v[i].end());
    }
}

void MatrixMath::stabilizeMatrix(vec2d & v){
    for(unsigned int i = 0; i < v.size(); i++){
        for(unsigned int j = 0; j < v[i].size(); j++){
            if(v[i][j] == 0)
                v[i][j] = std::numeric_limits<long double>::epsilon();
        }
    }
}

void MatrixMath::rescaleMatrix(vec2d & v, long double minVal, long double maxVal){
    const long double minValSrc = minMatrix(v);
    const long double maxValSrc = maxMatrix(v);
    const long double a = (maxVal - minVal)/(maxValSrc - minValSrc);
    const long double b = minVal - a * minValSrc;

    for(unsigned int i = 0; i < v.size(); i++){
        for(unsigned int j = 0; j < v[i].size(); j++){
                v[i][j] = a*v[i][j] + b;
        }
    }

}

auto MatrixMath::meansMatrixByColumns(const vec2d & v) -> vec {
    vec result(v[0].size(), 0);

    for(unsigned int i = 0; i < v[0].size(); i++){
        long double mean = 0;
        for(unsigned int j = 0; j < v.size(); j++){
            mean += v[j][i];
        }
        mean = mean / v.size();
        result[i] = mean;
    }

    return result;
}

auto MatrixMath::minMatrixByColumns(const vec2d & v) -> vec {
    vec res(v[0].size(), 0);

    for(unsigned int i = 0; i < v[0].size(); i++){
        vec temp(v.size(), 0);
        for(unsigned int j = 0; j < v.size(); j++){
            temp[j] = v[j][i];
        }
        res[i] = *std::min_element(temp.begin(), temp.end());
    }

    return res;
}

auto MatrixMath::maxMatrixByColumns(const vec2d & v) -> vec {
    vec res(v[0].size(), 0);

    for(unsigned int i = 0; i < v[0].size(); i++){
        vec temp(v.size(), 0);
        for(unsigned int j = 0; j < v.size(); j++){
            temp[j] = v[j][i];
        }
        res[i] = *std::max_element(temp.begin(), temp.end());
    }

    return res;
}

void MatrixMath::normalizeMatrixByColumns(vec2d & v){
    vec means = meansMatrixByColumns(v);
    vec mins = minMatrixByColumns(v);
    vec maxs = maxMatrixByColumns(v);
    for(unsigned int i = 0; i < v.size(); i++){
        for(unsigned int j = 0; j < v[i].size(); j++){
            v[i][j] = (v[i][j] - means[j]);//(maxs[j] - mins[j]);
        }
    }
}

long double MatrixMath::minMatrix(const MatrixMath::vec2d & v){
    long double minVal = std::numeric_limits<long double>::max();

    for(unsigned int i = 0; i < v.size(); i++){
        for(unsigned int j = 0; j < v[i].size(); j++){
            if(v[i][j] < minVal)
                minVal = v[i][j];
        }
    }

    return minVal;
}

long double MatrixMath::maxMatrix(const MatrixMath::vec2d & v){
    long double maxVal = std::numeric_limits<long double>::min();

    for(unsigned int i = 0; i < v.size(); i++){
        for(unsigned int j = 0; j < v[i].size(); j++){
            if(v[i][j] > maxVal)
                maxVal = v[i][j];
        }
    }

    return maxVal;
}

void MatrixMath::fftVector(vec & frame, unsigned int NFFT) {
    // resize frame to N
    // either append zeros or turncate samples
    if(NFFT > frame.size()){
        MatrixMath::vec zeros(NFFT - frame.size(), 0);
        frame.insert(frame.end(), zeros.begin(), zeros.end());
    }
    else if(NFFT < frame.size()){
        frame.erase(frame.begin() + NFFT, frame.end());
    }

    // "copy" vec into complex vector
    std::valarray<std::complex<long double>> complexFrame(frame.size());
    for(unsigned int i = 0; i < frame.size(); i++)
        complexFrame[i] = frame[i];

    // fourier transform
    MatrixMath::fft(complexFrame);

    // compute magnitude of FFT
    frame.clear();
    frame.resize(NFFT/2+1); // only left half of spectrum
    for(unsigned int i = 0; i < frame.size(); i++){
        frame[i] = sqrt(complexFrame[i].real() * complexFrame[i].real() + complexFrame[i].imag() * complexFrame[i].imag());
    }
}

void MatrixMath::fftMatrix(MatrixMath::vec2d & frames, unsigned int NFFT) {
    for(unsigned int i = 0; i < frames.size(); i++){
        fftVector(frames[i], NFFT);
    }
}

void MatrixMath::dctVector(vec & v){
    vec result(v.size(), 0);

    for(unsigned int i = 0; i < v.size(); i++)
        result[0] += v[i];
    result[0] *= 1 / sqrt(v.size());

    for(unsigned int i = 1; i < result.size(); i++){
        for(unsigned int j = 0; j < v.size(); j++){
            result[i] += v[j]*cos(3.14159265358979323846264338328L*i*(2*j+1)/(2*v.size()));
        }
        result[i] *= sqrt(2 / static_cast<long double>(v.size()));
    }

    v = std::move(result);
}

void MatrixMath::dctMatrix(vec2d & m){
    for(unsigned int i = 0; i < m.size(); i++){
        dctVector(m[i]);
    }
}

auto MatrixMath::linspace(long double low, long double high, unsigned int numPoints) -> MatrixMath::vec {
    MatrixMath::vec result(numPoints, 0);

    long double diff = high - low;
    long double step = diff / numPoints;

    for(unsigned int i = 0; i < numPoints; i++){
        result[i] = low + i*step;

        if(i == numPoints - 1)// rounding may occur here
            result[i] = high;
    }

    return result;
}

bool AudioProcessor::validateConfig() const {
    if(conf.bytesPerSample == 0 || conf.bytesPerSample > 2)
        return 0;
    if(conf.numberOfChannels == 0)
        return 0;
    if(conf.sampleRate == 0)
        return 0;
    if(conf.framingSize == 0)
        return 0;
    if(conf.framingStride == 0)
        return 0;
    if(conf.NFFT == 0)
        return 0;
    if(conf.numberOfFilterBanks == 0)
        return 0;
    if(conf.MFCC && conf.firstMFCC > conf.numberOfFilterBanks)
        return 0;
    if(conf.MFCC && conf.lastMFCC > conf.numberOfFilterBanks)
        return 0;
    if(conf.MFCC && conf.firstMFCC > conf.lastMFCC)
        return 0;
    if(conf.MFCC && conf.sinLift && conf.cepLifter < 1)
        return 0;
    if(conf.rescale && conf.rescaleMax == conf.rescaleMin)
        return 0;
    return 1;
}

auto AudioProcessor::bytesToSamples(const byteVec & buffer) const -> MatrixMath::vec {

    if(buffer.size() % conf.bytesPerSample * conf.numberOfChannels){
        throw AudioProcessorException("Invalid size of input audio buffer.");
    }

    MatrixMath::vec samples(buffer.size()/conf.bytesPerSample, 0);
    for(unsigned int i = 0, j = 0; i < buffer.size(); i += conf.bytesPerSample, j++){
        uint32_t sample = 0;
        for(unsigned int k = 0; k < conf.bytesPerSample; k++){
            //sample |= static_cast<unsigned int>((static_cast<unsigned char>(buffer[i + k]) << (8 * k)));
            sample |= (static_cast<uint8_t>(buffer[i + k]) << (8 * k));
        }
        if(conf.bytesPerSample == 1){
            samples[j] = static_cast<uint8_t>(sample);
        }
        else if(conf.bytesPerSample == 2){
            samples[j] = static_cast<int16_t>(sample);
        }
    }

    return samples;
}

void AudioProcessor::channelsToMono(MatrixMath::vec & sampleData) const {

    MatrixMath::vec samplesMono(sampleData.size()/conf.numberOfChannels,0);

    for(unsigned int i = 0, j = 0; i < sampleData.size(); i+=conf.numberOfChannels, j++){
        long double sumSignals = 0;
        for(unsigned int k = 0; k < conf.numberOfChannels; k++){
            sumSignals += sampleData[i + k];
        }
        samplesMono[j] = sumSignals / static_cast<long double>(conf.numberOfChannels);
    }

    sampleData = std::move(samplesMono);
}

void AudioProcessor::magnitudeToPower(MatrixMath::vec2d & magnitudes) {
    for(unsigned int i = 0; i < magnitudes.size(); i++){
        for(unsigned int j = 0; j < magnitudes[i].size(); j++){
            unsigned int N = magnitudes[i].size();
            magnitudes[i][j] = magnitudes[i][j] * magnitudes[i][j] / N;
        }
    }
}

auto AudioProcessor::frameSamples(MatrixMath::vec & sampleData) const -> MatrixMath::vec2d {

    const unsigned int frameLength = static_cast<unsigned int>(round(conf.framingSize / static_cast<long double>(1000) * conf.sampleRate)); // in num samples
    const unsigned int frameStep = static_cast<unsigned int>(round(conf.framingStride / static_cast<long double>(1000) * conf.sampleRate)); // in num samples
    const unsigned int numFrames = static_cast<unsigned int>(ceil((sampleData.size() - frameLength) / static_cast<long double>(frameStep)));

    // make sure there is valid number of samples to perform framing
    // if there is not then add zeros to the end to make it valid
    const int paddingToAppend = (sampleData.size() - frameLength) % frameStep;
    if(paddingToAppend){
        MatrixMath::vec zeros(paddingToAppend, 0);
        sampleData.insert(sampleData.end(), zeros.begin(), zeros.end());
    }

    MatrixMath::vec2d frames(numFrames, MatrixMath::vec(frameLength, 0));
    for(unsigned int i = 0, j = 0; i < numFrames; i++, j+=frameStep){
        std::copy(sampleData.begin() + j, sampleData.begin() + j + frameLength, frames[i].begin());
    }

    return frames;
}

void AudioProcessor::preEmphasis(MatrixMath::vec & sampleData) const {
    for(unsigned int i = 1; i < sampleData.size(); i++){
        sampleData[i] = sampleData[i] - conf.emphasisCoeff * sampleData[i - 1];
    }
}

void AudioProcessor::hammingWindow(MatrixMath::vec2d & frames) const {
    for(unsigned int i = 0; i < frames.size(); i++){
        for(unsigned int j = 0; j < frames[i].size(); j++){
            frames[i][j] *= 0.54L - 0.46L*cos((2*3.14159265358979323846264338328L*j)/static_cast<long double>((frames[i].size()-1)));
        }
    }
}

void AudioProcessor::filterBanks(MatrixMath::vec2d & v) const {
    const long double lowFreqMel = 0;
    const long double highFreqMel = hzToMel(conf.sampleRate / 2);
    MatrixMath::vec points = MatrixMath::linspace(lowFreqMel, highFreqMel, conf.numberOfFilterBanks + 2); // mel points equally spaced
    melToHz(points); // convert mel space into hz space

    for(unsigned int i = 0; i < points.size(); i++)
        points[i] = floor((conf.NFFT + 1) * points[i] / conf.sampleRate);

    MatrixMath::vec2d fBank(conf.numberOfFilterBanks, MatrixMath::vec(static_cast<int>(floor(conf.NFFT / 2 + 1)), 0));

    for(unsigned int i = 1; i < conf.numberOfFilterBanks + 1; i++){
        unsigned int fMinus = static_cast<int>(points[i - 1]);
        unsigned int f = static_cast<int>(points[i]);
        unsigned int fPlus = static_cast<int>(points[i + 1]);

        for(unsigned int j = fMinus; j < f; j++)
            fBank[i - 1][j] = (j - points[i - 1]) / (points[i] - points[i - 1]);
        for(unsigned int j = f; j < fPlus; j++)
            fBank[i - 1][j] = (points[i + 1] - j) / (points[i + 1] - points[i]);
    }

    MatrixMath::transposeMatrix(fBank);
    MatrixMath::dotMatrix(v, fBank);
    MatrixMath::stabilizeMatrix(v);

    //convert result to dB
    for(unsigned int i = 0; i < v.size(); i++){
        for(unsigned int j = 0; j < v[i].size(); j++){
            v[i][j] = 20 * log10(v[i][j]);
        }
    }
}

void AudioProcessor::sinLiftMatrix(MatrixMath::vec2d & v) const {
    MatrixMath::vec liftRow(v[0].size());
    for(unsigned int i = 0; i < liftRow.size(); i++){
        liftRow[i] = 1 + (conf.cepLifter / 2L) * sin(3.14159265358979323846264338328L * i / static_cast<long double>(conf.cepLifter));
    }

    for(unsigned int i = 0; i < v.size(); i++){
        for(unsigned int j = 0; j < v[i].size(); j++){
            v[i][j]*=liftRow[j];
        }
    }
}

auto AudioProcessor::processBuffer(const byteVec & buffer) const -> MatrixMath::vec2d {
    if(!validateConfig()){
        throw AudioProcessorException("Invalid audio configuration.");
    }

    // firstly, concatenate single bytes into audio samples
    MatrixMath::vec vectorData = bytesToSamples(buffer);

    // now translate all channel data into mono signal by using average of samples from all channels
    channelsToMono(vectorData);

    // apply pre emphasis filter to amplify high frequencies and increase s/n ratio
    preEmphasis(vectorData);

    // split audio samples into frames as frequencies are stationary over short periods of time
    // used to get good frequency contours of the signal
    MatrixMath::vec2d matrixData = frameSamples(vectorData);

    // apply hamming window to each frame to reduce spectral leakage
    hammingWindow(matrixData);

    // get frequency domain data from each frame
    MatrixMath::fftMatrix(matrixData, conf.NFFT);

    // convert magnitude to power spectrum
    magnitudeToPower(matrixData);

    // apply triangular filters on Mel scale to extract frequency bands
    filterBanks(matrixData);

    // apply MFCC if necessary
    if(conf.MFCC){
        MatrixMath::dctMatrix(matrixData);

        // erase not needed coeffs
        unsigned int numFromStart = conf.firstMFCC-1;
        unsigned int numToEnd = matrixData[0].size() - conf.lastMFCC;
        MatrixMath::eraseColumnsMatrix(matrixData, numFromStart, numToEnd);

        if(conf.sinLift){
            sinLiftMatrix(matrixData);
        }
    }

    if(conf.normalize)
        MatrixMath::normalizeMatrixByColumns(matrixData);

    MatrixMath::transposeMatrix(matrixData);

    if(conf.rescale){
        MatrixMath::rescaleMatrix(matrixData, conf.rescaleMin, conf.rescaleMax);
    }

    return matrixData;
}
