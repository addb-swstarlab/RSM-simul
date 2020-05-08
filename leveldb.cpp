#include "leveldb.h"
#include "util.h"
#include <fstream>
#include <sstream>
#include <iterator>
#include <string>
#include <sys/time.h>

#include <gsl/gsl_sf_erf.h>
#include <gsl/gsl_statistics.h>
#include <gsl/gsl_sort.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_cdf.h>

#include "policy_rl/Trainer.h"

// #define REMEMBER_NEXT_FIRST_KEY
//
LevelDB::LevelDB(const LevelDBParams& params, std::vector<Stat>& stats, Trainer* trainer)
    : params_(params), stats_(stats) {
  log_bytes_ = 0;
  // for log and level-0 that do not use compact()
  for (auto i = stats_.size(); i < 2; i++) stats_.push_back(Stat());

  levels_.push_back(sstables_t());

  level_bytes_.push_back(0);
  level_bytes_threshold_.push_back(
      static_cast<uint64_t>(-1));  // level-0 can accept any SSTable size

  if (params_.compaction_mode == LevelDBCompactionMode::kLinear)
    level_next_compaction_key_.push_back(LevelDBKeyMax);
  else if (params_.compaction_mode == LevelDBCompactionMode::kLinearNextFirst)
    level_next_compaction_key_.push_back(LevelDBKeyMin);

  inserts_ = 0;
  level_overflows_.push_back(0);
  level_compactions_.push_back(0);
  level_overlapping_sstables_.push_back(0);
  level_overlapping_sstables_false_.push_back(0);
  level_sweeps_.push_back(0);
  
  for(int i = 0; i < 4; i++) compaction_number.push_back(0);

  next_version_ = 0;
  compaction_id_ = 0;
  RSMtrainer_ = trainer;  
  read_bytes_non_output_ = 0;
  write_bytes_ = 0;
  srand(time(NULL));
}

LevelDB::~LevelDB() {
  for (std::size_t level = 0; level < levels_.size(); level++)
    for (auto& sstable : levels_[level]) delete sstable;
}

void LevelDB::print_status() const {
  printf("log: %zu items, %lu bytes\n", log_.size(), log_bytes_);
  for (std::size_t i = 0; i < levels_.size(); i++) {
    double overlaps = 0.;
    double overlaps_false = 0.;
    if (level_compactions_[i] != 0) {
      overlaps = level_overlapping_sstables_[i] /
                 static_cast<double>(level_compactions_[i]);
      overlaps_false = level_overlapping_sstables_false_[i] /
                       static_cast<double>(level_compactions_[i]);
    }
    uint64_t interval = 0;
    if (level_sweeps_[i] > 0) interval = inserts_ / level_sweeps_[i];
    printf(
        "level-%zu: %5zu tables, %14lu bytes, %6lu overflows, %6lu "
        "compactions, %5.2lf avg overlaps (%.2lf false), %4lu sweeps "
        "(interval=%8lu)\n",
        i, levels_[i].size(), level_bytes_[i], level_overflows_[i],
        level_compactions_[i], overlaps, overlaps_false, level_sweeps_[i],
        interval);
  }
}

void LevelDB::setCompaction(LevelDBCompactionMode mode) {
  params_.compaction_mode = mode;
}

void LevelDB::dump_state(FILE* fp) const {
  // XXX: Memtable is not dumped now.
  fprintf(fp, "next_version:%lu\n", next_version_);

  fprintf(fp, "log:\n");
  dump_state(fp, log_);

  fprintf(fp, "levels:\n");
  for (std::size_t level = 0; level < levels_.size(); level++) {
    auto& sstables = levels_[level];
    fprintf(fp, "level:\n");
    for (std::size_t i = 0; i < sstables.size(); i++) {
      fprintf(fp, "sstable:\n");
      dump_state(fp, *sstables[i]);
    }
  }
}

void LevelDB::dump_state(FILE* fp, const sstable_t& l) {
  for (std::size_t i = 0; i < l.size(); i++) dump_state(fp, l[i]);
}

void LevelDB::dump_state(FILE* fp, const LevelDBItem& item) {
#ifdef LEVELDB_TRACK_VERSION
  fprintf(fp, "item:%u,%lu,%u,%s\n", item.key, item.version,
          item.size & LevelDBItemSizeMask,
          item.size == LevelDBItemDeletion ? "T" : "F");
#else
  fprintf(fp, "item:%u,0,%u,%s\n", item.key, item.size & LevelDBItemSizeMask,
          item.size == LevelDBItemDeletion ? "T" : "F");
#endif
}

void LevelDB::put(LevelDBKey key, uint32_t item_size) {
#ifdef LEVELDB_TRACK_VERSION
  LevelDBItem item{key, item_size, next_version_++};
#else
  LevelDBItem item{key, item_size};
#endif
  inserts_++;
  append_to_log(item);
}

void LevelDB::del(LevelDBKey key) {
#ifdef LEVELDB_TRACK_VERSION
  LevelDBItem item{key, LevelDBItemDeletion, next_version_++};
#else
  LevelDBItem item{key, LevelDBItemDeletion};
#endif
  append_to_log(item);
}

uint64_t LevelDB::get(LevelDBKey key) {
  // TODO: Implement
  (void)key;
  return 0;
}

void LevelDB::force_compact() {
  flush_log();

  for (std::size_t level = 0; level < levels_.size() - 1; level++) {
    std::vector<std::vector<std::size_t>> sstable_indices;
    sstable_indices.push_back(std::vector<std::size_t>());
    sstable_indices.back().push_back(0);
    while (levels_[level].size() > 0) {
      compact(level, sstable_indices);
    }
  }
}

void LevelDB::append_to_log(const LevelDBItem& item) {
  log_.push_back(item);

  // Update statistics.
  auto new_log_bytes = log_bytes_ + item.size;
  // auto log_bytes_d = log_bytes_ / 4096;
  // auto new_log_bytes_d = new_log_bytes / 4096;
  // if (log_bytes_d != new_log_bytes_d) {
  //     // New blocks are written.
  //     stat_.write((new_log_bytes_d - log_bytes_d) * 4096);
  // }
  stats_[0].write(item.size);
  log_bytes_ = new_log_bytes;

  if (log_bytes_ > params_.log_size_threshold) flush_log();
}

void LevelDB::flush_log() {
  if (log_.size() == 0) return;

  // Simplified for simulation; a new SSTable is created from the memtable,
  // causing no disk read.
  sort_items(log_);
  levels_t sstable_runs;
  sstable_runs.push_back(sstables_t());
  sstable_runs.back().push_back(&log_);
  merge_sstables(sstable_runs, 0);
  delete_log();

  // TODO: LevelDB computes the score of each level: [current table count /
  // compaction trigger] (for level = 0) or [current level byte size / max level
  // byte size] (for level >= 1).
  //       It picks a level of the highest score in VersionSet::Finalize()
  //       (db/version_set.cc).
  //       Our checking is fine because compaction here is done synchronously
  //       and lower levels tend to get a higher score until being compacted.
  for (std::size_t level = 0; level < levels_.size(); level++)
    check_compaction(level);
}

void LevelDB::delete_log() {
  // stat_.del(log_bytes_ / 4096 * 4096);
  stats_[0].del(log_bytes_);
  log_.clear();
  log_bytes_ = 0;
}

struct _LevelDBKeyComparer {
  bool operator()(const LevelDBItem& a, const LevelDBItem& b) const {
    return a.key < b.key;
  }
};

void LevelDB::sort_items(sstable_t& items) {
  std::stable_sort(items.begin(), items.end(), _LevelDBKeyComparer());
}

struct _LevelDBSSTableComparer {
  LevelDB::sstable_t** sstables;
  std::size_t* sstables_pos;

  bool operator()(const std::size_t& a, const std::size_t& b) const {
    auto& item_a = (*sstables[a])[sstables_pos[a]];
    auto& item_b = (*sstables[b])[sstables_pos[b]];
    // Since std::make_heap makes a max-heap, we use a comparator with the
    // opposite result.
    if (item_a.key > item_b.key)
      return true;
    else if (item_a.key == item_b.key && a > b)
      return true;
    return false;
  }
};

void LevelDB::merge_sstables(const levels_t& sstable_runs, std::size_t level) {
  // The current SSTable in each run.
  std::size_t sstables_idx[sstable_runs.size()];
  sstable_t* sstables[sstable_runs.size()];

  // The current item in each run's current SSTable.
  std::size_t sstables_pos[sstable_runs.size()];

  for (std::size_t i = 0; i < sstable_runs.size(); i++) {
    assert(sstable_runs[i].size() != 0);
    sstables_idx[i] = 0;
    sstables[i] = sstable_runs[i][sstables_idx[i]];
    sstables_pos[i] = 0;
  }

  // Initialize push.
  push_state state;
  push_init(state, level);

  // Initialize a heap.
  std::vector<std::size_t> heap;
  _LevelDBSSTableComparer comp{sstables, sstables_pos};
  sequence(sstable_runs.size(), heap);
  std::make_heap(heap.begin(), heap.end(), comp);

  while (heap.size() != 0) {
    // Get the smallest key's SSTable index.
    auto i = heap.front();
    std::pop_heap(heap.begin(), heap.end(), comp);
    heap.pop_back();

    // Discover how many keys we can take from this SSTable.
    sstable_t* sstable = sstables[i];
    std::size_t size = sstable->size();

    std::size_t start = sstables_pos[i];
    std::size_t end;
    if (heap.size() == 0)
      // No other SSTables; we can take the remaining items in this SSTable.
      end = size;
    else {
      // Get the next smallest key's SSTable index (besides i's).
      auto j = heap.front();
      LevelDBKey next_possible_key = (*sstables[j])[sstables_pos[j]].key;

      end = start + 1;
      while (end < size && (*sstable)[end].key < next_possible_key) end++;
    }

    push_items(state, *sstable, start, end);

    if (end < size) {
      // More items in this SSTable.
      sstables_pos[i] = end;

      heap.push_back(i);
      std::push_heap(heap.begin(), heap.end(), comp);
    } else {
      // No more items in this SSTable.  Select the next SSTable in the same
      // run.
      sstables_idx[i]++;
      if (sstables_idx[i] < sstable_runs[i].size()) {
        sstables[i] = sstable_runs[i][sstables_idx[i]];
        sstables_pos[i] = 0;

        heap.push_back(i);
        std::push_heap(heap.begin(), heap.end(), comp);
      } else {
        // all SSTables in the same run have been consumed.
      }
    }
  }

  push_flush(state);
}

double get_time() {
  struct timeval tv_now;
  gettimeofday(&tv_now, NULL);

  return (double)tv_now.tv_sec + (double)tv_now.tv_usec/1000000UL;
}

void link_vertex(std::vector<float> &matrix, int src, int dst) {
  int size = sqrt(matrix.size());
  matrix[size*src + dst] = 1;
  matrix[size*dst + src] = 1;  
}

void LevelDB::up_propagate(std::size_t level) {
  if(level == 1) return;
  
  auto& level_tables = levels_[level];
  auto& level_tables_next = levels_[level - 1];  
  
  std::size_t sstable_idx_start = 0;
  std::size_t sstable_idx_end = 0;
    
  for(std::size_t k = 0; k < level_idx[level].size(); k++) {
    auto& sstable = *level_tables[level_idx[level][k].curr_idx];
    if (sstable_idx_end > 0) sstable_idx_start = sstable_idx_end - 1;
    
    while (sstable_idx_start < level_tables_next.size() && 
            level_tables_next[sstable_idx_start]->back().key < 
            sstable.front().key) sstable_idx_start++;
    
    sstable_idx_end = sstable_idx_start;
    
    while (sstable_idx_end < level_tables_next.size() &&
            level_tables_next[sstable_idx_end]->front().key < 
            sstable.back().key) sstable_idx_end++;

    if(sstable_idx_end - sstable_idx_start != 0) {    
      for(std::size_t i = sstable_idx_start; i < sstable_idx_end; i++) {
        if((level_idx[level-1].size() != 0) && 
         (level_idx[level-1].back().curr_idx == i)) continue;
        
        Fsize temp {0, i, 0, 0};
        level_idx[level-1].push_back(temp);
      }
    }
    
    level_idx[level][k].up_start_idx = sstable_idx_start;
    level_idx[level][k].up_end_idx = sstable_idx_end;
  }  
  up_propagate(level - 1);    
}
/*  std::size_t comp; // compare
    std::size_t curr_idx; // file idx
    std::size_t start_idx;
    std::size_t end_idx;
*/

void LevelDB::down_propagate(std::size_t level) { 
  if(level + 1 >= levels_.size()) return;
  
  auto& level_tables = levels_[level];
  auto& level_tables_next = levels_[level + 1];  
  
  std::size_t sstable_idx_start = 0;
  std::size_t sstable_idx_end = 0;
    
  for(std::size_t k = 0; k < level_idx[level].size(); k++) {
    auto& sstable = *level_tables[level_idx[level][k].curr_idx];
    if (sstable_idx_end > 0) sstable_idx_start = sstable_idx_end - 1;
    
    while (sstable_idx_start < level_tables_next.size() && 
            level_tables_next[sstable_idx_start]->back().key < 
            sstable.front().key) sstable_idx_start++;
    
    sstable_idx_end = sstable_idx_start;   
    while (sstable_idx_end < level_tables_next.size() &&
            level_tables_next[sstable_idx_end]->front().key < 
            sstable.back().key) sstable_idx_end++;

    if(sstable_idx_end - sstable_idx_start != 0) {    
      for(std::size_t i = sstable_idx_start; i < sstable_idx_end; i++) {
        if((level_idx[level+1].size() != 0) && 
         (level_idx[level+1].back().curr_idx == i)) continue;
        
        Fsize temp {0, i, 0, 0};
        level_idx[level+1].push_back(temp);
      }
    }

    level_idx[level][k].bot_start_idx = sstable_idx_start;
    level_idx[level][k].bot_end_idx = sstable_idx_end;
  }  
  down_propagate(level+1);  
}

torch::Tensor LevelDB::set_submatrix(std::size_t level, int* max_size) {
  for(uint i = 0; i < level_idx.size(); i++) level_idx[i].clear();
  level_idx.clear();
  for(uint i = 0; i < 5; i++) level_idx.push_back(std::vector<Fsize>());
  
  adj_matrix.clear();
    
//  std::cout << "DRAW GRAPH" << std::endl;
  /* Select best victims to draw graph */
  auto& level_tables = levels_[level];
  std::vector<Fsize> temp(level_tables.size());
  for (size_t i = 0; i < level_tables.size(); i++) {
    auto& sstable = *level_tables[i];
    temp[i].curr_idx = i;
    int diff = (sstable.back().key - sstable.front().key);
    temp[i].comp = (int) (diff/sstable.size());
  }
  
//  std::cout << "SORT BEST VICTIM" << std::endl;
  /* Sort best victims */
  std::sort(temp.begin(), temp.end(), 
            [](const Fsize& f1, const Fsize& f2) -> bool {
              return f1.comp < f2.comp;});
        
//  std::cout << "LEVEL SORT BEST VICTIM" << std::endl;
  num_victim = level_tables.size() < num_victim ? level_tables.size() : num_victim;
  for(uint i = 0; i < num_victim; i++) {
    level_idx[level].push_back(temp[i]);
  }
   
  std::sort(level_idx[level].begin(), level_idx[level].end(),
            [](const Fsize& f1, const Fsize& f2) -> bool {
              return f1.curr_idx < f2.curr_idx;});
      
//  std::cout << "DOWN PROPAGATE" << std::endl;
  down_propagate(level);
//  std::cout << "UP PROPAGATE" << std::endl;
  up_propagate(level);
  
  /*  int comp; // compare
    int curr_idx; // file idx
    int start_idx;
    int end_idx;
  */
  
  int maximum_size = 0;
  for (uint i = 0; i < levels_.size(); i++ ) {
    maximum_size += level_idx[i].size();
  }
  adj_matrix.resize(maximum_size*maximum_size);

//  for(uint i = 0; i < level_idx.size(); i++) {
//    for(uint j = 0; j < level_idx[i].size(); j++) {
//      std::cout << "[" << i << "][" << j << "] "<< level_idx[i][j].curr_idx << " && " 
//                << level_idx[i][j].bot_start_idx << " && "
//                << level_idx[i][j].bot_end_idx << " && " 
//                << level_idx[i][j].up_end_idx << " && " 
//                << level_idx[i][j].up_end_idx << std::endl;
//    }
//  } 
 
  int pre_idx = 0;
  int idx = 0; 
  for (uint i = level; i < levels_.size()-1; i++) {
    idx += level_idx[i].size();  
    std::size_t idx_next = 0;
    
    for (uint j = 0; j < level_idx[i].size(); j++) {
      int start_idx = level_idx[i][j].bot_start_idx;
      int end_idx = level_idx[i][j].bot_end_idx;

      /* required to modify --> while */
      for(uint k = idx_next; k < level_idx[i+1].size(); k++) {
        if(level_idx[i+1][k].curr_idx >= end_idx) {
          idx_next = k;
          break;
        }
        link_vertex(adj_matrix, pre_idx + j, idx + k);
      }
    }  
    pre_idx += level_idx[i].size();  
  }
  
  idx += level_idx[levels_.size()-1].size();
  
  pre_idx = 0;
  for (uint i = level; i > 0; i--) {
    std::size_t idx_next = 0;
    
    for (uint j = 0; j < level_idx[i].size(); j++) {
      int start_idx = level_idx[i][j].up_start_idx;
      int end_idx = level_idx[i][j].up_end_idx;

      int base = (i == level) ? j : pre_idx + j;
      
      /* required to modify --> while */
      for(uint k = idx_next; k < level_idx[i-1].size(); k++) {
        if(level_idx[i-1][k].curr_idx >= end_idx) {
          idx_next = k;
          break;
        }
        link_vertex(adj_matrix, base, idx + k);
      }
    } 
    pre_idx = idx;
    idx += level_idx[i-1].size();        
  }
  
  /* I matrix */
  for (int i = 0; i < maximum_size; i++) adj_matrix[maximum_size*i + i] = 1;
  
//  for(uint i = 0; i < maximum_size; i++) {
//    for(uint j = 0; j < maximum_size; j++) {
//      std::cout << "[" << i << "][" << j << "] : " << adj_matrix[maximum_size*i + j] << std::endl;
//    }
//  } 

  *max_size = maximum_size;
  return torch::from_blob(adj_matrix.data(), {1, maximum_size, maximum_size}, torch::dtype(torch::kFloat)).clone().detach();
}


torch::Tensor LevelDB::set_subfeature(std::size_t level, std::size_t maximum_size) {  
  feat_matrix.clear(); 
  feat_matrix.resize(3*maximum_size);
  
  uint64_t idx = 0;      
  float max_value = 0;
  max_value = log((float)params_.hint_num_unique_keys);
    
  for (uint i = level; i < levels_.size(); i++) {
    auto& level_tables = levels_[i];
    for (uint j = 0; j < level_idx[i].size(); j++) {
      auto& sstable = *level_tables[level_idx[i][j].curr_idx];

      if(sstable.front().key == 0) feat_matrix[idx++] = 0; 
      else feat_matrix[idx++] = log((float)sstable.front().key) / max_value;
      feat_matrix[idx++] = log((float)sstable.back().key) / max_value;
      feat_matrix[idx++] = (float)sstable.size();
    }  
  }
  
  for (uint i = level - 1; i > 0; i--) {
    auto& level_tables = levels_[i];
    for (uint j = 0; j < level_idx[i].size(); j++) {
      auto& sstable = *level_tables[level_idx[i][j].curr_idx];

      if(sstable.front().key == 0) feat_matrix[idx++] = 0; 
      else feat_matrix[idx++] = log((float)sstable.front().key) / max_value;
      feat_matrix[idx++] = log((float)sstable.back().key) / max_value;
      feat_matrix[idx++] = (float)sstable.size();
    }  
  }
  
  uint64_t max = 0;
  for(std::size_t j = 0; j < idx/3; j++) {
    if(max < feat_matrix[3*j + 2])
      max = feat_matrix[3*j + 2];   
  }
  
  for(std::size_t j = 0; j < idx/3; j++)
    feat_matrix[3*j + 2] /= max; 

  return torch::from_blob(feat_matrix.data(), {1, (long int)maximum_size, 3}, torch::dtype(torch::kFloat)).clone().detach();
}

std::size_t LevelDB::select_action(std::size_t level) {
//  std::size_t act_idx = (int) (RSMtrainer_->Action[0] * (num_victim - 1)); 
  std::size_t act_idx = (int) (RSMtrainer_->Action_DQN); 
  std::size_t selected = level_idx[level][act_idx].curr_idx;  
  if(compaction_id_ % 1000 == 0) {
    std::cout << std::setprecision(32);
    std::cout << "ACTION [" << level << "] : " << RSMtrainer_->Action_DQN << std::endl;
    std::cout << "SELECTED = " << selected <<std::endl; 
  }
  return selected;
}

void LevelDB::check_compaction(std::size_t level) {
  if (level == 0) {
    // Compact if we have too many level-0 SSTables.
    if (levels_[0].size() >= params_.level0_sstable_count_threshold) {
      level_overflows_[0]++;
      level_sweeps_[0]++;
      std::vector<std::vector<std::size_t>> sstable_indices;
      for (std::size_t i = 0; i < levels_[0].size(); i++) {
        sstable_indices.push_back(std::vector<std::size_t>());
        sstable_indices.back().push_back(i);
      }
      compact(0, sstable_indices);
      assert(levels_[0].size() == 0);
    }
  } else {
    // Compact if we have too much data in this level.
    if (level_bytes_[level] > level_bytes_threshold_[level]) {
      level_overflows_[level]++;
      std::vector<std::vector<std::size_t>> sstable_indices;
      sstable_indices.push_back(std::vector<std::size_t>());

      while (level_bytes_[level] > level_bytes_threshold_[level]) {
        sstable_indices.back().clear();
        
        if (params_.compaction_mode == LevelDBCompactionMode::kLinear ||
            params_.compaction_mode ==
                LevelDBCompactionMode::kLinearNextFirst) {
          // Find the next table to compact.
          auto& level_tables = levels_[level];
          std::size_t count = level_tables.size();
          std::size_t i;
          for (i = 0; i < count; i++) {
            auto& sstable = *level_tables[i];

            if (params_.compaction_mode == LevelDBCompactionMode::kLinear) {
              if (sstable.front().key > level_next_compaction_key_[level])
                break;
            } else if (params_.compaction_mode ==
                       LevelDBCompactionMode::kLinearNextFirst) {
              if (sstable.front().key >= level_next_compaction_key_[level])
                break;
            }
          }
          if (i == count) {
            i = 0;
            level_sweeps_[level]++;
          }
          if (params_.compaction_mode == LevelDBCompactionMode::kLinear) {
            level_next_compaction_key_[level] = level_tables[i]->back().key;
          } else if (params_.compaction_mode ==
                     LevelDBCompactionMode::kLinearNextFirst) {
            if (i < count - 1)
              level_next_compaction_key_[level] =
                  level_tables[i + 1]->front().key;
            else
              level_next_compaction_key_[level] = LevelDBKeyMax;
          }
          sstable_indices.back().push_back(i);
        } else if (params_.compaction_mode ==
                   LevelDBCompactionMode::kMostNarrow) {
          auto& level_tables = levels_[level];
          std::size_t count = level_tables.size();

          // TODO: This is quite slow -- O(N).  We may probably want to make it
          // O(logN) with a priority queue.
          std::size_t selected = count;
          LevelDBKey min_width = 0;
          for (std::size_t i = 0; i < count; i++) {
            auto& sstable = *level_tables[i];
            LevelDBKey width = sstable.back().key - sstable.front().key;
            if (selected == count || min_width > width) {
              min_width = width;
              selected = i;
            }
          }
          assert(selected != count);
          sstable_indices.back().push_back(selected);
        } else if (params_.compaction_mode ==
                   LevelDBCompactionMode::kLeastOverlap) {
          auto& level_tables = levels_[level];
          std::size_t count = level_tables.size();

          if (level < levels_.size() - 1) {
            // TODO: This is quite slow -- O(N).  We may probably want to make
            // it O(logN) with some magic (this is complicated because overlaps
            // change as we compact).
            auto& level_tables_next = levels_[level + 1];
            std::size_t selected = count;
            std::size_t min_overlap = 0;
            std::size_t sstable_idx_start = 0;
            std::size_t sstable_idx_end = 0;
            for (std::size_t i = 0; i < count; i++) {
              auto& sstable = *level_tables[i];
              if (sstable_idx_end > 0) sstable_idx_start = sstable_idx_end - 1;
              while (sstable_idx_start < level_tables_next.size() &&
                     level_tables_next[sstable_idx_start]->back().key <
                         sstable.front().key)
                sstable_idx_start++;
              sstable_idx_end = sstable_idx_start;
              while (sstable_idx_end < level_tables_next.size() &&
                     level_tables_next[sstable_idx_end]->front().key <
                         sstable.back().key)
                sstable_idx_end++;

              std::size_t overlap = sstable_idx_end - sstable_idx_start;
              // if (overlap != 0) {
              //     printf("range: [%u,%u]\n", sstable.front().key,
              //     sstable.back().key);
              //     printf("overlap: %zu[%u,%u] - %zu[%u,%u]\n",
              //     sstable_idx_start,
              //     level_tables_next[sstable_idx_start]->front().key,
              //     level_tables_next[sstable_idx_start]->back().key,
              //     sstable_idx_end - 1, level_tables_next[sstable_idx_end -
              //     1]->front().key, level_tables_next[sstable_idx_end -
              //     1]->back().key);
              // }
              if (selected == count || min_overlap > overlap) {
                min_overlap = overlap;
                selected = i;
              }
            }
            assert(selected != count);
            sstable_indices.back().push_back(selected);
          } else {
            // We cannot use find_overlapping_tables() if the next level is not
            // created yet.
            sstable_indices.back().push_back(0);
          }
        } else if (params_.compaction_mode ==
                   LevelDBCompactionMode::kLargestRatio) {
          auto& level_tables = levels_[level];
          std::size_t count = level_tables.size();

          if (level < levels_.size() - 1) {
            // TODO: This is quite slow -- O(N).  We may probably want to make
            // it O(logN) with some magic (this is complicated because overlaps
            // change as we compact).
            auto& level_tables_next = levels_[level + 1];
            std::size_t selected = count;
            double max_ratio = 0.;
            std::size_t sstable_idx_start = 0;
            std::size_t sstable_idx_end = 0;
            for (std::size_t i = 0; i < count; i++) {
              auto& sstable = *level_tables[i];
              if (sstable_idx_end > 0) sstable_idx_start = sstable_idx_end - 1;
              while (sstable_idx_start < level_tables_next.size() &&
                     level_tables_next[sstable_idx_start]->back().key <
                         sstable.front().key)
                sstable_idx_start++;
              sstable_idx_end = sstable_idx_start;
              while (sstable_idx_end < level_tables_next.size() &&
                     level_tables_next[sstable_idx_end]->front().key <
                         sstable.back().key)
                sstable_idx_end++;

              // TODO: Use LevelDBItem::size instead of the item count.
              std::size_t s = 0;
              for (std::size_t j = sstable_idx_start; j < sstable_idx_end; j++)
                s += level_tables_next[j]->size();
              // Make division cleaner.
              if (s == 0) s = 1;

              double ratio =
                  static_cast<double>(sstable.size()) / static_cast<double>(s);
              if (selected == count || max_ratio < ratio) {
                max_ratio = ratio;
                selected = i;
              }
            }
            assert(selected != count);
            sstable_indices.back().push_back(selected);
          } else {
            // We cannot use find_overlapping_tables() if the next level is not
            // created yet.
            sstable_indices.back().push_back(0);
          }
        } else if (params_.compaction_mode ==
                   LevelDBCompactionMode::kWholeLevel) {
          level_sweeps_[level]++;
          sequence(levels_[level].size(), sstable_indices.back());
        } else if (params_.compaction_mode ==
                   LevelDBCompactionMode::kRSMTrain ||
                   params_.compaction_mode == LevelDBCompactionMode::kRSMEvaluate) {        
          if(set_input) {
            /* case: input is set in previous step */
              
            std::vector<float> Reward;
            float waf = (float) read_bytes_non_output_ / (float) write_bytes_;
            Reward.push_back(waf); // 1/WAF
            
            std::vector<int64_t> Action;
            Action.push_back(RSMtrainer_->Action_DQN);
            
            if (params_.compaction_mode == LevelDBCompactionMode::kRSMTrain) {
              torch::Device device(torch::kCPU);
              int* max;

             // double start_matrix =  get_time();
              torch::Tensor post_adj_tensor = set_submatrix(level, max).to(device);
              torch::Tensor post_feat_tensor = set_subfeature(level, *max).to(device);  

              torch::Tensor action_tensor = torch::tensor(Action, torch::dtype(torch::kFloat)).to(device);
              torch::Tensor reward_tensor = torch::tensor(Reward, torch::dtype(torch::kFloat)).to(device);
              //std::cout << "matrix generation = " << get_time() - start_matrix << std::endl;

             // double start_push =  get_time();
              RSMtrainer_->buffer.push(prev_adj_tensor, prev_feat_tensor, 
                      post_adj_tensor, post_feat_tensor, action_tensor.unsqueeze(0), reward_tensor.unsqueeze(0));
             // std::cout << "push time = " << get_time() - start_push << std::endl;
            
              if(RSMtrainer_->buffer.size_buffer() >= 2000) {
                //double start_learn =  get_time();
                RSMtrainer_->learn();
                //std::cout << "learn time = " << get_time() - start_learn << std::endl;
              }
           
//              if(compaction_id_ % 10000 == 0) {
//                RSMtrainer_->saveCheckPoints(); 
//              }
            }
            
            RSMtrainer_->rewards_.emplace_back(Reward.at(0)); 
            set_input = false;
          }
          
          int* max;

         // double start_input =  get_time();
          prev_adj_tensor = set_submatrix(level,max);
          prev_feat_tensor = set_subfeature(level, *max);
         // std::cout << "input matrix generation = " << get_time() - start_input << std::endl;
        /* DDPG code */
        /*
          if(params_.compaction_mode == LevelDBCompactionMode::kRSMTrain) 
            RSMtrainer_->Action_DDPG = RSMtrainer_->act_ddpg(feat_matrix, adj_matrix, true); 
          else
            RSMtrainer_->Action_DDPG = RSMtrainer_->act_ddpg(feat_matrix, adj_matrix, false);
         * */
       //   double start_act =  get_time();
          RSMtrainer_->Action_DQN = RSMtrainer_->act_dqn(feat_matrix, adj_matrix); 
         // std::cout << "input matrix generation = " << get_time() - start_act << std::endl;
          std::size_t selected = select_action(level);
          sstable_indices.back().push_back(selected); 
          set_input = true;

          /* Random Action Code for copy/paste */
//          RSMtrainer_->Action.clear();
//          for(uint i = 0; i < action_size; i++) {
//            RSMtrainer_->Action.push_back((double)rand()/(double)RAND_MAX);
//          } 
          
        }             
        else
          assert(false);

        read_bytes_non_output_ = 0;
        write_bytes_ = 0;

        compact(level, sstable_indices);
        
        if(params_.compaction_mode == LevelDBCompactionMode::kRSMTrain ||
           params_.compaction_mode == LevelDBCompactionMode::kRSMEvaluate) {
          compaction_number[level-1]++;
          compaction_id_++;
        
          if(((compaction_id_-1) % 1000 == 0)) {
            std::cout << std::setprecision(32);
            std::cout << "insert = " << inserts_ << std::endl;
            std::cout << "level = " << level << " & compaction_id = " << compaction_id_ - 1<< std::endl;
            std::cout << "read = " << (float) read_bytes_non_output_ << " write = " << (float) write_bytes_ <<std::endl;
            std::cout << "Reward = " << ((float) read_bytes_non_output_/ (float) write_bytes_) <<std::endl;
          }
        }
        
      }
    }
  }
}

void LevelDB::push_init(push_state& state, std::size_t level) {
  state.level = level;

  state.pending_item = nullptr;

  state.current_sstable = nullptr;

  state.current_sstable_size = 0;
  state.use_split_key = false;
}

void LevelDB::push_items(push_state& state, const sstable_t& sstable,
                         std::size_t start, std::size_t end) {
  assert(start != end);

  bool level0 = (state.level == 0);
  bool last_level = (state.level == levels_.size() - 1);

  if (state.pending_item == nullptr) {
    state.pending_item = &sstable[start];
    start++;
  }

  while (start != end) {
    bool drop_pending_item = false;
    if (state.pending_item->size == LevelDBItemDeletion && last_level)
      drop_pending_item = true;
    else if (state.pending_item->key == sstable[start].key) {
#ifdef LEVELDB_TRACK_VERSION
      if (state.pending_item->version >= sstable[start].version)
        printf("pv %lu cv %lu level %zu start %zu end %zu\n",
               state.pending_item->version, sstable[start].version, state.level,
               start, end);
      assert(state.pending_item->version < sstable[start].version);
#endif
      drop_pending_item = true;
    }

    if (!drop_pending_item) {
      if (state.current_sstable == nullptr)
        state.current_sstable = new sstable_t();

      state.current_sstable->push_back(*state.pending_item);
      state.current_sstable_size +=
          state.pending_item->size & LevelDBItemSizeMask;

      if (state.current_sstable->size() == 1 && !params_.use_custom_sizes) {
        // Determine the split key; the current SSTable should not contain this
        // split key, otherwise it will overlap with too many SSTables in the
        // next level.
        if (level0 || last_level)
          state.use_split_key = false;
        else {
          auto& level_tables = levels_[state.level + 1];
          std::size_t count = level_tables.size();

          std::size_t i;
          // Choose the first SSTable in the next level that can potentially
          // overlap.
          // TODO: Use binary search and memorization from previous run.
          for (i = 0; i < count; i++) {
            auto& sstable = *level_tables[i];
            if (state.pending_item->key <= sstable.back().key) break;
          }
          // XXX: This follows LevelDB's impl.html, but the actual
          // implementation uses bytes instead of the number of SSTables.
          //      See kMaxGrandParentOverlapBytes (db/version_set.cc).
          std::size_t end =
              std::min(i + params_.sstable_overlap_threshold, count);
          if (end < count) {
            // Remember the split key.
            state.use_split_key = true;
            state.split_key = level_tables[end]->front().key;
          } else {
            // Splitting by key will never happen because there will be few
            // overlapping tables.
            state.use_split_key = false;
          }
        }
      }
    }

    state.pending_item = &sstable[start];

    bool need_new_sstable = false;
    if (state.use_split_key && state.pending_item->key >= state.split_key)
      need_new_sstable = true;
    else {
      uint64_t item_size = state.pending_item->size & LevelDBItemSizeMask;
      // Level-0 generates only one SSTable per merge. Otherwise, we obey the
      // maximum SSTable size.
      if (!level0 &&
          state.current_sstable_size + item_size >
              params_.sstable_size_threshold)
        need_new_sstable = true;
    }

    if (need_new_sstable) {
      if (state.current_sstable != nullptr) {
        state.current_sstable->shrink_to_fit();
        state.completed_sstables.push_back(state.current_sstable);
        level_bytes_[state.level] += state.current_sstable_size;
        stats_[1 + state.level].write(state.current_sstable_size);
        write_bytes_ += state.current_sstable_size;

        state.current_sstable = nullptr;

        state.current_sstable_size = 0;
        state.use_split_key = false;
      }
    }

    start++;
  }
}

void LevelDB::push_flush(push_state& state) {
  // printf("push_flush level %zu\n", state.level);
  bool level0 = (state.level == 0);
  bool last_level = (state.level == levels_.size() - 1);

  // Flush the pending item.
  if (state.pending_item != nullptr) {
    bool drop_pending_item = false;
    if (state.pending_item->size == LevelDBItemDeletion && last_level)
      drop_pending_item = true;

    if (!drop_pending_item) {
      if (state.current_sstable == nullptr) {
        state.current_sstable = new sstable_t();
        state.current_sstable_size = 0;
      }

      state.current_sstable->push_back(*state.pending_item);
      state.current_sstable_size +=
          state.pending_item->size & LevelDBItemSizeMask;
    }
  }

  // Flush the current SSTable.
  if (state.current_sstable != nullptr) {
    state.current_sstable->shrink_to_fit();
    state.completed_sstables.push_back(state.current_sstable);
    level_bytes_[state.level] += state.current_sstable_size;
    stats_[1 + state.level].write(state.current_sstable_size);
    write_bytes_ += state.current_sstable_size;
  }

  // Insert new SSTables into the level.
  if (level0)
    levels_[0].insert(levels_[0].end(), state.completed_sstables.begin(),
                      state.completed_sstables.end());
  else {
    auto& level_tables = levels_[state.level];
    std::size_t count = level_tables.size();

    std::size_t i;
    for (i = 0; i < count; i++) {
      auto& sstable = *level_tables[i];
      if (state.pending_item->key <= sstable.back().key) break;
    }

    level_tables.insert(
        std::next(level_tables.begin(), static_cast<std::ptrdiff_t>(i)),
        state.completed_sstables.begin(), state.completed_sstables.end());
  }
}

void LevelDB::find_overlapping_tables(
    std::size_t level, const LevelDBKey& first, const LevelDBKey& last,
    std::vector<std::size_t>& out_sstable_indices) {
  assert(level >= 1);
  assert(level < levels_.size());

  // TODO: Use binary search to reduce the search range.

  auto& level_tables = levels_[level];
  std::size_t count = level_tables.size();
  out_sstable_indices.clear();

  for (std::size_t i = 0; i < count; i++) {
    auto& sstable = *level_tables[i];
    if (!(last < sstable.front().key || sstable.back().key < first))
      out_sstable_indices.push_back(i);
  }
}

void LevelDB::compact(
    std::size_t level,
    const std::vector<std::vector<std::size_t>>& sstable_indices) {
  // printf("compact level %zu\n", level);
  // Ensure we have all necessary data structures for the next level.
  if (levels_.size() <= level + 1) {
    levels_.push_back(sstables_t());
    level_bytes_.push_back(0);

    for (auto i = stats_.size(); i < 2 + level + 1; i++)
      stats_.push_back(Stat());
    level_overflows_.push_back(0);
    level_compactions_.push_back(0);
    level_overlapping_sstables_.push_back(0);
    level_overlapping_sstables_false_.push_back(0);
    level_sweeps_.push_back(0);

    // E.g., level_size for level-1  = params_.level_size_ratio
    // E.g., level_size for level-2  = params_.level_size_ratio *
    // params_.growth_factor
    uint64_t level_size = params_.level_size_ratio;
    for (std::size_t i = 1; i < level + 1; i++)
      level_size *= params_.growth_factor;

    if (params_.use_custom_sizes) {
      level_size = 0;
      std::ifstream ifs("output_sensitivity.txt");
      while (!ifs.eof()) {
        std::string line;
        std::getline(ifs, line);

        std::istringstream iss(line);
        std::vector<std::string> tokens{std::istream_iterator<std::string>{iss},
                                        std::istream_iterator<std::string>{}};

        if (tokens.size() < 4) continue;
        if (tokens[0] != "sensitivity_item_count_leveldb_best_sizes" &&
            tokens[0] != "sensitivity_log_size_leveldb_best_sizes")
          continue;
        if (static_cast<uint64_t>(atol(tokens[1].c_str())) !=
            params_.hint_num_unique_keys)
          continue;
        if (atof(tokens[2].c_str()) != params_.hint_theta) continue;
        if (static_cast<uint64_t>(atol(tokens[3].c_str())) !=
            params_.log_size_threshold)
          continue;

        assert(level < tokens.size() - 5);
        // Assume the item size of 1000 bytes.
        level_size = static_cast<uint64_t>(
            atof(tokens[5 + level].c_str()) * 1000. + 0.5);
        break;
      }
      assert(level_size != 0);
    }
    printf("level-%zu: max size %lu bytes\n", level + 1, level_size);
    level_bytes_threshold_.push_back(level_size);

    if (params_.compaction_mode == LevelDBCompactionMode::kLinear ||
        params_.compaction_mode == LevelDBCompactionMode::kLinearNextFirst)
      level_next_compaction_key_.push_back(LevelDBKeyMax);
  }

  // Discover SSTables to merge.
  std::vector<std::size_t> sstable_indices_current;
  for (auto& sstable_indices_sub : sstable_indices)
    for (auto i : sstable_indices_sub) sstable_indices_current.push_back(i);

  std::vector<std::size_t> sstable_indices_next;
  LevelDBKey min_key;
  LevelDBKey max_key;
  if (params_.compaction_mode == LevelDBCompactionMode::kLinear ||
      params_.compaction_mode == LevelDBCompactionMode::kLinearNextFirst ||
      params_.compaction_mode == LevelDBCompactionMode::kMostNarrow ||
      params_.compaction_mode == LevelDBCompactionMode::kLeastOverlap ||
      params_.compaction_mode == LevelDBCompactionMode::kLargestRatio ||
      params_.compaction_mode == LevelDBCompactionMode::kRSMTrain || 
      params_.compaction_mode == LevelDBCompactionMode::kRSMEvaluate ) {
    min_key = LevelDBKeyMax;
    max_key = LevelDBKeyMin;
    for (auto i : sstable_indices_current) {
      min_key = std::min(min_key, levels_[level][i]->front().key);
      max_key = std::max(max_key, levels_[level][i]->back().key);
    }
    find_overlapping_tables(level + 1, min_key, max_key, sstable_indices_next);
  } else if (params_.compaction_mode == LevelDBCompactionMode::kWholeLevel) {
    min_key = LevelDBKeyMin;
    max_key = LevelDBKeyMax;
    sequence(levels_[level + 1].size(), sstable_indices_next);
  } else
    assert(false);

  // level_compactions_[level] += sstable_indices_current.size();
  // level_overlapping_sstables_[level] += sstable_indices_next.size();

  // level_compactions_[level]++;
  // level_overlapping_sstables_[level] +=
  // static_cast<double>(sstable_indices_next.size()) /
  // static_cast<double>(sstable_indices_current.size());

  // TODO: Use LevelDBItem::size instead of the item count.
  uint64_t s0 = 0;
  uint64_t s1 = 0;
  uint64_t s1_false = 0;
  for (auto i : sstable_indices_current) s0 += levels_[level][i]->size();
  for (auto i : sstable_indices_next) s1 += levels_[level + 1][i]->size();
  for (auto i : sstable_indices_next)
    for (auto& item : *levels_[level + 1][i])
      if (item.key < min_key || item.key > max_key) s1_false++;
  level_compactions_[level]++;
  level_overlapping_sstables_[level] +=
      static_cast<double>(s1) / static_cast<double>(s0);
  level_overlapping_sstables_false_[level] +=
      static_cast<double>(s1_false) / static_cast<double>(s0);

  // printf("overlapping\n");
  // printf("  level %zu (%zu):", level, levels_[level].size());
  // for (auto i : sstable_indices_current)
  // 	printf(" %zu", i);
  // printf("\n  level %zu (%zu):", level + 1, levels_[level + 1].size());
  // for (auto i : sstable_indices_next)
  // 	printf(" %zu", i);
  // printf("\n");

  levels_t source_sstables;
  if (sstable_indices_next.size() != 0) {
    source_sstables.push_back(sstables_t());
    for (auto i : sstable_indices_next) {
      source_sstables.back().push_back(levels_[level + 1][i]);

      std::uint64_t sstable_size = 0;
      for (auto& item : *source_sstables.back().back())
        sstable_size += item.size & LevelDBItemSizeMask;
      level_bytes_[level + 1] -= sstable_size;
      stats_[1 + level + 1].read(sstable_size);
      stats_[1 + level + 1].del(sstable_size);
    }
  }
  for (auto& sstable_indices_sub : sstable_indices) {
    source_sstables.push_back(sstables_t());
    for (auto i : sstable_indices_sub) {
      source_sstables.back().push_back(levels_[level][i]);

      std::uint64_t sstable_size = 0;
      for (auto& item : *source_sstables.back().back())
        sstable_size += item.size & LevelDBItemSizeMask;
      level_bytes_[level] -= sstable_size;
      // We are reading from level, but let level+1 have the numbers to follow
      // the convention used in the analysis
      // stats_[1 + level].read(sstable_size);
      stats_[1 + level + 1].read(sstable_size);
      stats_[1 + level].del(sstable_size);
      read_bytes_non_output_ += sstable_size;
    }
  }

  {
    std::sort(sstable_indices_current.begin(), sstable_indices_current.end());
    std::reverse(sstable_indices_current.begin(),
                 sstable_indices_current.end());
    for (auto i : sstable_indices_current) remove_sstable(level, i);

    std::reverse(sstable_indices_next.begin(), sstable_indices_next.end());
    for (auto i : sstable_indices_next) remove_sstable(level + 1, i);
  }

  merge_sstables(source_sstables, level + 1);

  // Delete old SSTables.
  for (auto& sstables : source_sstables)
    for (auto& sstable : sstables) delete sstable;
}

LevelDB::sstable_t* LevelDB::remove_sstable(std::size_t level,
                                            std::size_t idx) {
  sstable_t* t = levels_[level][idx];

  for (auto j = idx; j < levels_[level].size() - 1; j++)
    levels_[level][j] = levels_[level][j + 1];
  levels_[level].pop_back();

  return t;
}
