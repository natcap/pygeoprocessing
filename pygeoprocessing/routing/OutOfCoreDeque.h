#ifndef _OUTOFCOREDEQUE_H_
#define _OUTOFCOREDEQUE_H_

#include <deque>

namespace pygeoprocessing {
    template <class T>
    class OutOfCoreDeque {
        public:
            OutOfCoreDeque();
            ~OutOfCoreDeque();
            int size();
            void push_front(const T& x);
            void push_back(const T& x);
            const T& front();
            const T& back();
            void pop_front();
            void pop_back();

        private:
            std::deque<T> deque;
    };
}

#include "OutOfCoreDeque.cpp"

#endif
