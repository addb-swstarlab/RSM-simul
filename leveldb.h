#pragma once

#include "common.h"
#include "stat.h"
#include <cstdio>
#include "policy_rl/Trainer.h"

// #define LEVELDB_TRACK_VERSION

typedef uint64_t LevelDBKey;
static const uint64_t LevelDBKeyMin = 0;
static const uint64_t LevelDBKeyMax = static_cast<uint64_t>(-1);

enum class LevelDBCompactionMode {
  // LevelDB's default compaction; pick one SSTable and pick the next linearly.
  kLinear = 0,
  // Similar to above but remember the first key of the next available SSTable
  // instead of the last key of the compacted SSTable.
  kLinearNextFirst = 1,
  // Pick an SSTable with the narrow key range.
  kMostNarrow = 2,
  // Pick an SSTable with the least number of next-level SSTables that overlap
  // with it.
  kLeastOverlap = 3,
  // Pick an SSTable whose size ratio to next-level overlapping SSTables size
  // (potential the inverse of write amplification) is the greatest; this is
  // similar to HyperLevelDB's strategy (see VersionSet::PickCompaction() in
  // HyperLevelDB/db/version_set.cc).
  kLargestRatio = 4,
  // Always compact the whole level (like LSM-tree).
  kWholeLevel = 5,

  // RocksDB - Pick an SSTable whose size is the maximum (default) + 1
  // compaction thread
  kRocksDBMaxSize = 6,
  // RocksDB - Pick an SSTable in the same way as kLinear + 1 compaction thread
  kRocksDBLinear = 7,
  // RocksDB - kRocksDBMaxSize + 4 compaction threads
  kRocksDBMaxSizeMT = 8,
  // RocksDB - kRocksDBLinear + 4 compaction threads
  kRocksDBLinearMT = 9,

  // RocksDB - Universal Compaction
  kRocksDBUniversal = 10,
  // RSM-tree Compaction
  kRSMTrain = 11,
  kRSMEvaluate = 12,
};

struct LevelDBParams {
  // When a log file exceeds this size, a new Level-0 SSTable is created, and a
  // new log file is created.
  uint64_t log_size_threshold;
  // When the level 0 ("young") has this many SSTables, all of them are merged
  // into the next level.
  uint64_t level0_sstable_count_threshold;
  // When an SSTable file exceeds this size, a new SSTable is created.
  uint64_t sstable_size_threshold;
  // When a level-L SSTable's key range overlaps with this many level-(L+1)
  // SSTables, a new level-L SSTable is created.
  uint64_t sstable_overlap_threshold;
  // When the level L is (growth factor)^L * (level size ratio) bytes big, an
  // level-L SSTable and all overlapping level-(L+1) SSTables are merged and
  // form new level-(L+1) SSTables.  The level-L SSTable is chosen in a
  // round-robin way.
  uint64_t growth_factor;
  // The size of level 1.
  uint64_t level_size_ratio;

  // The compaction mode.
  LevelDBCompactionMode compaction_mode;

  // Use custom level sizes.
  bool use_custom_sizes;
  // Hints used for custom_sizes
  uint64_t hint_num_unique_keys;
  double hint_theta;

  // Enable fsync for implementation-based tests.
  bool enable_fsync;

  LevelDBParams() {
    log_size_threshold =
        4 * 1048576;  // write_buffer_size (include/leveldb/options.h)
    level0_sstable_count_threshold =
        4;  // When LevelDB triggers compaction (db/dbformat.h)
    // level0_sstable_count_threshold = 8;     // When LevelDB slows down new
    // insertion
    // level0_sstable_count_threshold = 12;     // When LevelDB stops handling
    // new insertion
    sstable_size_threshold =
        2 * 1048576;  // kTargetFileSize (db/version_set.cc)
    sstable_overlap_threshold =
        10;              // kMaxGrandParentOverlapBytes (db/version_set.cc)
    growth_factor = 10;  // MaxBytesForLevel() (db/version_set.cc)
    level_size_ratio = 10 * 1048576;  // MaxBytesForLevel() (db/version_set.cc)

    use_custom_sizes = false;
    hint_num_unique_keys = 0;
    hint_theta = 0.;

    enable_fsync = false;
  }
};

struct LevelDBItem {
  LevelDBKey key;
  uint32_t size;
#ifdef LEVELDB_TRACK_VERSION
  uint64_t version;
#endif
};

static const uint32_t LevelDBItemSizeMask = 0x7fffffffU;
static const uint32_t LevelDBItemDeletion = 0x80000010U;

// A LevelDB simulation based on
// https://leveldb.googlecode.com/svn/trunk/doc/impl.html
class LevelDB {
 public:
  LevelDB(const LevelDBParams& params, std::vector<Stat>& stats);
  ~LevelDB();

  // Prints the summary of the store.
  void print_status() const;
  
  void print_network_status() const;
  
  void setCompaction(LevelDBCompactionMode mode);

  // Writes the current items in the store to the file.
  void dump_state(FILE* fp) const;

  // Puts a new item in the store.
  void put(LevelDBKey key, uint32_t item_size);

  // Deletes an item from the store.
  void del(LevelDBKey key);

  // Gets an item from the store.
  uint64_t get(LevelDBKey key);

  // Forces compaction until there is no SSTable except the last level.
  void force_compact();

  typedef std::vector<LevelDBItem> sstable_t;
  typedef std::vector<sstable_t*> sstables_t;
  typedef std::vector<sstables_t> levels_t;

  // typedef std::vector<LevelDBItem*> item_ptr_t;

 protected:
  // Adds a new item to the log.
  void append_to_log(const LevelDBItem& item);

  // Flushes all in-memory data to disk.  This effectively creates new level-0
  // SSTables from the Memtable.
  void flush_log();

  // Deletes the log.
  void delete_log();

  // Sorts items in place.
  void sort_items(sstable_t& items);

  // Merges SSTables and emits SSTable in the specified level.  Items at a later
  // position take precedence.
  void merge_sstables(const levels_t& source_sstables, std::size_t level);

  void set_initial();
  //void set_state(bool input);
  void set_state(std::vector<double> &state);
  
  std::size_t select_action(std::size_t level);
  
  // Check if we need new compaction.
  void check_compaction(std::size_t level);
  
  // Pushes items to a level, creating SSTables.
  struct push_state {
    std::size_t level;

    const LevelDBItem* pending_item;

    sstable_t* current_sstable;

    uint64_t current_sstable_size;
    bool use_split_key;
    LevelDBKey split_key;

    sstables_t completed_sstables;
  };
  void push_init(push_state& state, std::size_t level);
  void push_items(push_state& state, const sstable_t& sstable,
                  std::size_t start, std::size_t end);
  void push_flush(push_state& state);

  // Finds all overlapping SSTables in the level.
  void find_overlapping_tables(std::size_t level, const LevelDBKey& first,
                               const LevelDBKey& last,
                               std::vector<std::size_t>& out_sstable_indices);

  // Performs compaction with SSTables from the level and all over overlapping
  // SSTables in the next level.
  void compact(std::size_t level,
               const std::vector<std::vector<std::size_t>>& sstable_indices);

  // // Removes an SSTable from the level.  This does not release the memory
  // used by the SSTable.
  sstable_t* remove_sstable(std::size_t level, std::size_t idx);

  // Writes an item list to the file.
  static void dump_state(FILE* fp, const sstable_t& l);
  static void dump_state(FILE* fp, const LevelDBItem& item);

 private:
  LevelDBParams params_;
  std::vector<Stat>& stats_;
  sstable_t log_;
  uint64_t log_bytes_;
  levels_t levels_;
  std::vector<uint64_t> level_bytes_;
  std::vector<uint64_t> level_bytes_threshold_;
  // for LevelDBCompactionMode::kLinear and
  // LevelDBCompactionMode::kLinearNextFirst
  std::vector<LevelDBKey> level_next_compaction_key_;
  uint64_t inserts_;
  std::vector<uint64_t> level_overflows_;
  std::vector<uint64_t> level_compactions_;
  std::vector<double> level_overlapping_sstables_;
  std::vector<double> level_overlapping_sstables_false_;
  std::vector<uint64_t> level_sweeps_;
  uint64_t next_version_;
  uint64_t compaction_id_;
  Trainer* RSMtrainer_;
  uint64_t read_bytes_non_output_;
  uint64_t write_bytes_;
  int64_t level_size = 4;
  int64_t channel_size = 3;
  int64_t action_size = 1;
  int64_t bucket_size = 4096;
  std::vector<uint64_t> compaction_number;
  
};
