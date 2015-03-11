#include <deque>
#include "OutOfCoreDeque.h"

template <class T>
OutOfCoreDeque<T>::OutOfCoreDeque() {}

template <class T>
OutOfCoreDeque<T>::~OutOfCoreDeque() {}

template <class T>
int OutOfCoreDeque<T>::size() {
    this->deque.size();
}

template <class T>
void OutOfCoreDeque<T>::push_front(const T& x) {
    this->deque.push_front(x);
}

template <class T>
void OutOfCoreDeque<T>::push_back(const T& x) {
    this->deque.push_back(x);
}

template <class T>
const T& OutOfCoreDeque<T>::front() {
    return this->deque.front();
}

template <class T>
const T& OutOfCoreDeque<T>::back() {
    return this->deque.back();
}

template <class T>
void OutOfCoreDeque<T>::pop_front() {
    this->deque.pop_front();
}

template <class T>
void OutOfCoreDeque<T>::pop_back() {
    this->deque.pop_back();
}
