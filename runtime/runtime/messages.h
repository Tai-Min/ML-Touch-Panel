#ifndef MESSAGES_H
#define MESSAGES_H


#include <string>
using namespace std;

/**
 * @brief Write message in white color.
 * @param msg Message to write.
 */
void normal(string msg);

/**
 * @brief Write message in blue color.
 * @param msg Message to write.
 */
void info(string msg);

/**
 * @brief Write message in green color.
 * @param msg Message to write.
 */
void success(string msg);

/**
 * @brief Write message in yellow color.
 * @param msg Message to write.
 */
void warn(string msg);

/**
 * @brief Write message in red color.
 * @param msg Message to write.
 */
void error(string msg);

#endif // MESSAGES_H
