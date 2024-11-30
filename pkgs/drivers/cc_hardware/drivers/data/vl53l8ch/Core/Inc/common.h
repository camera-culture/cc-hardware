#ifndef __COMMON_H
#define __COMMON_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

typedef struct {
  uint16_t resolution;           // 16 or 64
  uint16_t ranging_mode;         // 0: Continuous, 1: Autonomous
  uint16_t ranging_frequency_hz; // Hz
  uint16_t integration_time_ms;  // ms
  uint16_t cnh_start_bin;
  uint16_t cnh_num_bins;
  uint16_t cnh_subsample;
  uint16_t agg_start_x;
  uint16_t agg_start_y;
  uint16_t agg_merge_x;
  uint16_t agg_merge_y;
  uint16_t agg_cols;
  uint16_t agg_rows;
} SensorConfig;

typedef enum __attribute__((__packed__)) {
  CONFIGURE = 0,
  STOP = 1,
  START = 2,
} CommandType;


typedef struct {
  CommandType type;
  SensorConfig config;
} Command;

#ifdef __cplusplus
}
#endif

#endif /* __COMMON_H */
