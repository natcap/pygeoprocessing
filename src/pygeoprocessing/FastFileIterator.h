#ifndef __FASTFILEITERATOR_H_INCLUDED__
#define __FASTFILEITERATOR_H_INCLUDED__

#include <cstddef>
#include <iostream>
#include <fstream>
#include <stdio.h>

using namespace std;
template <class DATA_T> class FastFileIterator{
private:
    DATA_T* buffer = nullptr;
    char* file_path = nullptr;
    // these offsets and sizes are in numbers of items instead of bytes, or
    // number of bytes / sizeof(DATA_T)
    size_t global_offset;
    size_t local_offset;
    size_t cache_size;
    size_t buffer_size;
    size_t file_length;

    void update_buffer() {
        if (this->local_offset >= this->cache_size) {
            this->global_offset += this->local_offset;
            this->local_offset = 0;
            if (this->buffer_size > (this->file_length - this->global_offset)) {
                this->cache_size = this->file_length - this->global_offset;
            } else {
                this->cache_size = this->buffer_size;
            }
            if (this->buffer != nullptr) {
                delete [] this->buffer;
            }
            this->buffer = new DATA_T[this->cache_size];
            FILE *fptr = fopen(this->file_path, "r");
            fseek(fptr, this->global_offset*sizeof(DATA_T), SEEK_SET);
            printf("reading cache size %d", this->cache_size);
            fread((DATA_T*)this->buffer, sizeof(DATA_T), this->cache_size, fptr);
            fclose(fptr);
        }
    }

public:
    FastFileIterator(const char *file_path, size_t buffer_size) {
        global_offset = 0;
        local_offset = 0;
        cache_size = 0;
        this->buffer_size = buffer_size;
        this->file_path = new char[strlen(file_path)+1];
        this->file_path = strcpy(this->file_path, file_path);
        std::ifstream is (this->file_path, std::ifstream::binary);
        is.seekg(0, is.end);
        this->file_length = is.tellg() / sizeof(DATA_T);
        printf("file length: %d buffer size: %d\n", this->file_length, this->buffer_size);
        if (this->buffer_size > this->file_length) {
            this->buffer_size = this->file_length;
        }
        printf("file length: %d buffer size: %d\n", this->file_length, this->buffer_size);
        is.close();
        update_buffer();
    }
    ~FastFileIterator() {
        if (buffer != nullptr) {
            delete [] buffer;
            delete [] this->file_path;
        }
    }

    DATA_T const peek() {
        return this->buffer[this->local_offset];
    }

    size_t const size() {
        return this->file_length - (this->global_offset + this->local_offset);
    }

    DATA_T next() {
        update_buffer();
        this->local_offset += 1;
        return this->buffer[this->local_offset-1];
    }
};

template <class DATA_T>
int FastFileIteratorCompare(FastFileIterator<DATA_T>* a,
                            FastFileIterator<DATA_T>* b) {
    return a->peek() < b->peek();
}

#endif
