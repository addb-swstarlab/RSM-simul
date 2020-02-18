#pragma once

#include "common.h"

class Stat {
 public:
  Stat() { reset_all(); }

  void reset() {
    read_count_ = 0;
    read_bytes_ = 0;
    write_count_ = 0;
    write_bytes_ = 0;
    delete_count_ = 0;
    delete_bytes_ = 0;
  }

  void reset_all() {
    reset();
    current_bytes_ = 0;
  }

  void read(uint64_t num_bytes) {
    read_count_++;
    read_bytes_ += static_cast<int64_t>(num_bytes);
  }

  void write(uint64_t num_bytes) {
    write_count_++;
    write_bytes_ += static_cast<int64_t>(num_bytes);
    current_bytes_ += static_cast<int64_t>(num_bytes);
  }

  void overwrite(uint64_t num_bytes) {
    write_count_++;
    write_bytes_ += static_cast<int64_t>(num_bytes);
  }

  void del(uint64_t num_bytes) {
    delete_count_++;
    delete_bytes_ += static_cast<int64_t>(num_bytes);
    current_bytes_ -= static_cast<int64_t>(num_bytes);
  }

  int64_t read_count() const { return read_count_; }
  int64_t read_bytes() const { return read_bytes_; }
  int64_t write_count() const { return write_count_; }
  int64_t write_bytes() const { return write_bytes_; }
  int64_t delete_count() const { return delete_count_; }
  int64_t delete_bytes() const { return delete_bytes_; }
  int64_t current_bytes() const { return current_bytes_; }

  void print_status() const {
    printf("Read: %ld times, %ld bytes\n", read_count_, read_bytes_);
    printf("Write: %ld times, %ld bytes\n", write_count_, write_bytes_);
    printf("Delete: %ld times, %ld bytes\n", delete_count_, delete_bytes_);
    printf("Current size: %ld bytes\n", current_bytes_);
  }

 private:
  int64_t read_count_;
  int64_t read_bytes_;
  int64_t write_count_;
  int64_t write_bytes_;
  int64_t delete_count_;
  int64_t delete_bytes_;
  int64_t current_bytes_;
};
