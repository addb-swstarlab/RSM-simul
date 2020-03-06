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

#include "policy_rl/DDPGTrainer.h"

// #define REMEMBER_NEXT_FIRST_KEY

LevelDB::LevelDB(const LevelDBParams& params, std::vector<Stat>& stats)
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

  next_version_ = 0;
  compaction_id_ = 0;
  RSMtrainer_ = new DDPGTrainer(8,2,10240, params_.model_load);  
  read_bytes_non_output_ = 0;
  write_bytes_ = 0;
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

void LevelDB::print_network_status() const {
  FILE* fp_reward = fopen("/home/wonki/rsm-simul/reward_info.txt", "wt");
  fprintf(fp_reward, " ==============Reward============== \n");

  for(uint i = 0; i < RSMtrainer_->rewards_.size(); i++) {
    fprintf(fp_reward, "%lf\n", RSMtrainer_->rewards_[i]);
  }
  
  FILE* fp_actor = fopen("/home/wonki/rsm-simul/actor_info.txt", "wt");
  
  fprintf(fp_actor, " ==============Actor Loss============== \n");
  for(uint i = 0; i < RSMtrainer_->actor_loss_.size(); i++) {
    fprintf(fp_actor, "%lf\n", RSMtrainer_->actor_loss_[i]);
  }
  
  FILE* fp_critic = fopen("/home/wonki/rsm-simul/critic_info.txt", "wt");
  fprintf(fp_critic, " ==============Critic Loss============== \n");
  for(uint i = 0; i < RSMtrainer_->critic_loss_.size(); i++) {
    fprintf(fp_critic, "%lf\n", RSMtrainer_->critic_loss_[i]);
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
  //std::cout << "insert num = " << inserts_ << std::endl;
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

double nrd0(double x[], const int N) {
  gsl_sort(x, 1, N);
  double hi = gsl_stats_sd(x, 1, N);
  double iqr = gsl_stats_quantile_from_sorted_data (x,1, N,0.75) - gsl_stats_quantile_from_sorted_data (x,1, N,0.25);
  double lo = GSL_MIN(hi, iqr/1.34);
  double bw = 0.9 * lo * pow(N,-0.2);
  return(bw);
}

double GaussKernel(double x) { 
  return exp(-(gsl_pow_2(x)/2))/(M_SQRT2*sqrt(M_PI)); 
}

double GaussCdf(double x) { 
  double cdf = (1 + gsl_sf_erf(x/M_SQRT2))/2; 
  return cdf;
}

double KernelDensity(double *samples, double obs, size_t n) {
  size_t i;
  double h = GSL_MAX(nrd0(samples, n), 1e-6);
  double prob = 0;
  for(i = 0; i < n; i++)
  {
    prob += GaussKernel((obs - samples[i])/h)/(n*h);
  }
  return prob;
}

double KernelCdf(double *samples, double obs, size_t n) {
  size_t i;
  //double h = GSL_MAX(nrd0(samples, n), 1e-6);
  //std::cout << " h = " << h << std::endl;
  double prob = 0;
  for(i = 0; i < n; i++)
  {
//    prob += GaussCdf((obs - samples[i])/h)/(n*h);
      prob += GaussCdf((obs - samples[i]))/(n);
  }
  return prob;
}

double get_time() {
  struct timeval tv_now;
  gettimeofday(&tv_now, NULL);

  return (double)tv_now.tv_sec + (double)tv_now.tv_usec/1000000UL;
}

void LevelDB::set_initial() {
  for(unsigned int i = 0; i < channel_size*level_size*256; i++) {
    double prob = 0.0;
    RSMtrainer_->PrevState.emplace_back(prob);    
  }
}

void LevelDB::set_state(bool input) {
//  uint64_t start_t;
  // start_t = get_time();
  //  std::cout << "clear elapsed time: " << 
//                (double)(get_time() - start_t) / 100 <<std::endl;
  if(input) {
    RSMtrainer_->PrevState.clear();
    RSMtrainer_->PrevState = RSMtrainer_->PostState;
    if(RSMtrainer_->PrevState.size() == 0) set_initial();  
    
//    std::cout << "==============>Prev<===============" <<std::endl;
//    for(unsigned int l = 0; l < channel_size; l++) {
//        std::cout <<"===========channel : " << l << "=============== " << std::endl;
//      for(unsigned int i = 1; i < 5; i++) {
//        double sum = 0.0;
//        for(unsigned int k = 0; k < 256; k++) {
//          std::cout << "[" << RSMtrainer_->PrevState[256*level_size*l + 256*(i-1) + k] <<"] ";  
//          sum += RSMtrainer_->PrevState[256*level_size*l + 256*(i-1) + k];
//        }
//        std::cout << "sum = " << sum <<std::endl;
//      }
//    }
    
  }  else {
    RSMtrainer_->PostState.clear();
    RSMtrainer_->PostState = RSMtrainer_->PrevState;
    
    /* initialize input-output level state*/
    for(unsigned int l = 0; l < channel_size; l++) {
      for(unsigned int i = 1; i < 5; i++) {
        for(unsigned int k = 0; k < 256; k++) {
          RSMtrainer_->PostState[256*level_size*l + 256*(i-1) + k] = 0;  
        }
      }
    }
    
    for(unsigned int i = 1; i < 5; i++) {
      if(i > levels_.size() - 1) break;
      /* sample */
      int comp = (*levels_[i][0]).size() * levels_[i].size();
      int samples_per_level = comp < 100 ? comp : comp/100;

      int cnt_per_file = samples_per_level / levels_[i].size();
      if(cnt_per_file == 0) {
        cnt_per_file++;
        samples_per_level = levels_[i].size();
      }

      for(unsigned int j = 0; j < levels_[i].size(); j++) {
        std::vector<LevelDBItem> sst = *levels_[i][j];
        int item_per_cnt = (sst.size() / cnt_per_file);
        int traverse = 0;
        if (item_per_cnt == 0 ) {
          traverse = sst.size();
          item_per_cnt++;
        } else {
          traverse = cnt_per_file;
        }
 
        for(unsigned int k = 0; k < traverse; k++) {
          for(unsigned int l = 0; l < channel_size; l++) {
            int channel = (int)(sst[item_per_cnt*k].key / (double) pow(256, channel_size - 1 - l)) % 256; 
            RSMtrainer_->PostState[256*level_size*l + 256*(i-1) + channel] += (1/(double)samples_per_level);
          }  
        }      
      }
    }
    
//    std::cout << "==============>Post<===============" <<std::endl;
//    for(unsigned int l = 0; l < channel_size; l++) {
//      std::cout <<"===========channel : " << l << "=============== " << std::endl;
//      for(unsigned int i = 1; i < 5; i++) {
//        double sum = 0.0;
//        for(unsigned int k = 0; k < 256; k++) {
//          std::cout << "[" << RSMtrainer_->PostState[256*level_size*l + 256*(i-1) + k] <<"] ";  
//          sum += RSMtrainer_->PostState[256*level_size*l + 256*(i-1) + k];
//        }
//        std::cout << "sum = " << sum <<std::endl;
//      }
//    }
  }
}

//std::size_t LevelDB::select_action(std::size_t level) {
//  auto& level_tables = levels_[level];
//  std::size_t count = level_tables.size();
//  std::size_t selected = 0;
//  double min = std::numeric_limits<double>::max();
//  int factor = 384;
//  //int factor = 1;
//  
//  for (std::size_t i = 0; i < count; i++) {
//    auto& sstable = *level_tables[i];
//    double val = 0.0;
//
//    for(unsigned int j = 0; j < action_size; j++) {
//      int comp = (int) (RSMtrainer_->Action.at(j) * 256);
//      double min_range = ((int)((double)sstable.front().key / (double)pow(256, channel_size - 1 - j))) % 256;  
//      double max_range = ((int)((double)sstable.back().key / (double)pow(256, channel_size - 1 - j))) % 256;
//      double multiplier = (channel_size/2 - 1 - j) >= 0 ? pow(factor, channel_size/2 - 1 - j) : (1/(pow(factor, j - channel_size/2 + 1)));
//      val += (sqrt(pow(min_range - comp, 2) + pow(max_range - comp, 2)) * multiplier);
//      //val += sqrt(pow(min_range - comp, 2) + pow(max_range - comp, 2));
////      std::cout << "pow = " << (double)pow(256, 8 -1 - j) << " and " << (int)((double)sstable.front().key / (double)pow(256, 8 -1 - j)) <<std::endl;
////     std::cout << "sstable front = " << sstable.front().key << " sstable back = " << sstable.back().key <<std::endl;
//     // std::cout << " action[" << j << "] = " << comp << std::endl; 
////      std::cout << "multiplier = " << multiplier << " action[" << j << "] = " << comp << " min_range : " << min_range << " max_range = " << max_range << " val = " << val << std::endl;
//    }
//    //std::cout << std::setprecision(16) <<std::endl;
////    if(compaction_id_ % 1000 == 0) {
////      std::cout << std::setprecision(32) <<std::endl;
////      for(unsigned int j = 0; j < action_size; j++) {
////        double min_range = ((int)((double)sstable.front().key / (double)pow(256, channel_size - 1 - j))) % 256;  
////        double max_range = ((int)((double)sstable.back().key / (double)pow(256, channel_size - 1 - j))) % 256;
//// 
////        std::cout << "j = " << j << " min : " << min_range << " max : " << max_range << std::endl;
////      }
////      std::cout << "total [" << i << "] val = " << val << std::endl;
////    }
//    
//    if(val < min) {
//      min = val;
//      selected = i;
//    }
//  }
//  if(compaction_id_ % 1000 == 0) {
//    for(unsigned int j = 0; j < action_size; j++) 
//      std::cout << "Action [" << j << "] : " << RSMtrainer_->Action.at(j) << " and " << (int) (RSMtrainer_->Action.at(j) * 256) << std::endl;
//    std::cout << "count = " << count << " selected = " << selected <<std::endl; 
//  }
//  return selected;
//}

std::size_t LevelDB::select_action(std::size_t level) {
  auto& level_tables = levels_[level];
  std::size_t count = level_tables.size();
  std::size_t selected = 0;
  
  for (std::size_t i = 0; i < count; i++) {
    auto& sstable = *level_tables[i];

    int byte = (int) (RSMtrainer_->Action.at(0) * 7);
    double max_value = 0;
    for (int i = 0; i < byte + 1; i++) max_value += 255 * pow(256, i);
    double value = RSMtrainer_->Action.at(1) * max_value;
    
    double min = (double)sstable.front().key ;  
    double max = (double)sstable.back().key;
      
    if(min <= value < max) {
      selected = i;
    }
  }
  if(compaction_id_ % 1000 == 0) {
    std::cout << "Action [" << 0 << "] : " << (int) (RSMtrainer_->Action.at(0) * 7) << std::endl;
    std::cout << "Action [" << 1 << "] : " << (RSMtrainer_->Action.at(1)) << std::endl;
    double test_value = 0;
    for (int i = 0; i < (int) (RSMtrainer_->Action.at(0) * 7) + 1; i++) test_value += 255 * pow(256, i);
    double value = RSMtrainer_->Action.at(1) * test_value;

    std::cout << "test_value = " << test_value << " and value = " << value << std::endl;
    std::cout << "count = " << count << " selected = " << selected <<std::endl; 
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
                   LevelDBCompactionMode::kRSMPolicy) {        
          set_state(true); // input state
//          if(compaction_id_ < 4096) {
//            srand((unsigned int)time(NULL));
//            RSMtrainer_->Action.clear();
//            for(int i = 0; i < action_size; i++) 
//              RSMtrainer_->Action.push_back((double) rand() / (RAND_MAX));  
//          } else {
            if(!params_.model_load)  RSMtrainer_->Action = RSMtrainer_->act(RSMtrainer_->PrevState, true);
            else RSMtrainer_->Action = RSMtrainer_->act(RSMtrainer_->PrevState, false);
          //}
          std::size_t selected = select_action(level);
          sstable_indices.back().push_back(selected); 
        }             
        else
          assert(false);

        read_bytes_non_output_ = 0;
        write_bytes_ = 0;

        compact(level, sstable_indices);
        
        if (params_.compaction_mode ==
                   LevelDBCompactionMode::kRSMPolicy) {
          compaction_id_++;

          if(compaction_id_ % 1000 == 0) {
            std::cout << "level = " << level << " and compaction_id = " << compaction_id_ << std::endl;
            //std::cout << "Reward = " << (double)(1 / ((double)write_bytes_ / (double)read_bytes_non_output_)) <<std::endl;
          }
          std::vector<double> Reward;
          Reward.push_back((double)read_bytes_non_output_ / (double)write_bytes_ ); // 1/WAF
          set_state(false);

          if (!params_.model_load) {
            torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
            torch::Tensor state_tensor = torch::from_blob(RSMtrainer_->PrevState.data(), {1, channel_size, level_size, 256}, torch::dtype(torch::kDouble)).to(device);
            torch::Tensor new_state_tensor = torch::from_blob(RSMtrainer_->PostState.data(), {1, channel_size, level_size, 256}, torch::dtype(torch::kDouble)).to(device);
      
            std::vector<double> tempAction;
            if( RSMtrainer_->Action.size() == 0 ) {
              std::cout << "not here" <<std::endl;
              for(int i = 0; i < action_size; i++) tempAction.push_back(0); // we does not consider level = 0;
            } else {
              tempAction = RSMtrainer_->Action;
            }
            
            torch::Tensor action_tensor = torch::tensor(tempAction, torch::dtype(torch::kDouble)).to(device);
            torch::Tensor reward_tensor = torch::tensor(Reward, torch::dtype(torch::kDouble)).to(device);

            RSMtrainer_->buffer.push(state_tensor, new_state_tensor, action_tensor.unsqueeze(0), reward_tensor.unsqueeze(0));

            if(RSMtrainer_->buffer.size_buffer() >= 2048) {
              RSMtrainer_->learn();
            }
           
            if(compaction_id_ % 10000 == 0) {
              RSMtrainer_->saveCheckPoints(); 
            }
          }

          if (compaction_id_ % 100 == 0) {
            RSMtrainer_->rewards_.push_back(Reward.at(0));  
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
      params_.compaction_mode == LevelDBCompactionMode::kRSMPolicy) {
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
