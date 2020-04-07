// Copyright 2019 The Natural Capital Project
#ifndef SRC_PYGEOPROCESSING_FASTFILEITERATORINDEX_H_
#define SRC_PYGEOPROCESSING_FASTFILEITERATORINDEX_H_

#include <stdio.h>
#include <cstddef>
#include <iostream>
#include <fstream>

template <class DATA_T> class FastFileIteratorIndex{
 private:
    DATA_T* buffer = NULL;
    long long* index_buffer = NULL;
    char* file_path = NULL;
    char* index_path = NULL;
    DATA_T last_val = -1;
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
            if (this->buffer != NULL) {
                free(this->buffer);
                free(this->index_buffer);
            }
            this->buffer = reinterpret_cast<DATA_T*>(malloc(
                this->cache_size * sizeof(DATA_T)));
            this->index_buffer = reinterpret_cast<long long*>(malloc(
                this->cache_size * sizeof(long long)));
            FILE *fptr = fopen(this->file_path, "rb");
            fseek(fptr, this->global_offset * sizeof(DATA_T), SEEK_SET);
            FILE *index_fptr = fopen(this->index_path, "rb");
            size_t elements_to_read = this->cache_size;
            size_t elements_read = 0;
            while (elements_to_read) {
                fread(
                    reinterpret_cast<void*>(
                        this->index_buffer+elements_read*sizeof(long long)),
                    sizeof(long long), elements_to_read, index_fptr);
                elements_read += fread(
                    reinterpret_cast<void*>(
                        this->buffer+elements_read*sizeof(DATA_T)),
                    sizeof(DATA_T), elements_to_read, fptr);

                if (ferror(fptr)) {
                    perror("error occured");
                    elements_to_read = 0;
                } else if (feof(fptr)) {
                    printf("end of file\n");
                    break;
                } else {
                    elements_to_read = this->cache_size - elements_read;
                }
            }
            fclose(fptr);
            fclose(index_fptr);
        }
    }

 public:
    FastFileIteratorIndex(
            const char *file_path, const char *index_path, size_t buffer_size) {
        global_offset = 0;
        local_offset = 0;
        cache_size = 0;
        this->buffer_size = buffer_size;
        this->file_path = reinterpret_cast<char*>(malloc(
            (strlen(file_path)+1)*sizeof(char)));
        strncpy(this->file_path, file_path, strlen(file_path)+1);
        this->index_path = reinterpret_cast<char*>(malloc(
            (strlen(index_path)+1)*sizeof(char)));
        strncpy(this->index_path, index_path, strlen(index_path)+1);
        std::ifstream is(this->file_path, std::ifstream::binary);
        is.seekg(0, is.end);
        this->file_length = is.tellg() / sizeof(DATA_T);
        if (this->buffer_size > this->file_length) {
            this->buffer_size = this->file_length;
        }
        is.close();
        update_buffer();
    }
    ~FastFileIteratorIndex() {
        if (this->buffer != NULL) {
            free(this->buffer);
            free(this->file_path);
            free(this->index_path);
        }
    }

    DATA_T const peek() {
        return this->buffer[this->local_offset];
    }

    size_t const size() {
        return this->file_length - (this->global_offset + this->local_offset);
    }

    DATA_T get_last_val() {
        return this->last_val;
    }

    long long next() {
        if (size() > 0) {
            this->last_val = this->buffer[this->local_offset];
            long long index = this->index_buffer[this->local_offset];
            this->local_offset += 1;
            update_buffer();
            return index;
        } else {
            return -1;
        }
    }
};

template <class DATA_T>
int FastFileIteratorIndexCompare(FastFileIteratorIndex<DATA_T>* a,
                            FastFileIteratorIndex<DATA_T>* b) {
    return a->peek() < b->peek();
}

#endif  // SRC_PYGEOPROCESSING_FASTFILEITERATORINDEX_H_
