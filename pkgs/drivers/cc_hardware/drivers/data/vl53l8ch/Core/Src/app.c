#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "common.h"
#include "vl53lmz_api.h"
#include "vl53lmz_plugin_cnh.h"

// Sensor Configuration
VL53LMZ_Configuration Dev;
VL53LMZ_Motion_Configuration cnh_config;
uint32_t cnh_data_size;

// Function prototypes
uint8_t process_config(SensorConfig config);

// Main function renamed to a more relevant name
int app(volatile uint8_t *command_ready, volatile SensorConfig *sensor_config) {

  /*********************************/
  /*   VL53LMZ ranging variables   */
  /*********************************/

  uint8_t status;
  VL53LMZ_ResultsData Results; /* Results data from VL53LMZ */

  cnh_data_buffer_t cnh_data_buffer;

  int agg_id, bin_num;
  float amb_value, bin_value;

  int32_t *p_hist = NULL;
  int8_t *p_hist_scaler = NULL;
  int32_t *p_ambient = NULL;
  int8_t *p_ambient_scaler = NULL;

  /*********************************/
  /*      Customer platform        */
  /*********************************/

  Dev.platform.address = VL53LMZ_DEFAULT_I2C_ADDRESS;

  /*********************************/
  /*   Power on sensor and init    */
  /*********************************/

  /* Process the config */
  if (process_config(*sensor_config) != VL53LMZ_STATUS_OK) {
    printf("Error: process_config failed\n");
    return status;
  }

  printf("VL53LMZ ULD ready! (Version: %s)\n", VL53LMZ_API_REVISION);

  /*********************************/
  /*   Start UART Reception        */
  /*********************************/

  while (1) {
    if (*command_ready) {
      printf("Applying sensor configuration...\n");
      *command_ready = 0;
      if (process_config(*sensor_config) != VL53LMZ_STATUS_OK) {
        printf("Error: process_config failed\n");
        return status;
      }
      continue;
    }

    uint8_t isReady = 0;
    status = vl53lmz_check_data_ready(&Dev, &isReady);
    if (isReady) {
      vl53lmz_get_ranging_data(&Dev, &Results);

      /* Extract and process CNH data */
      status = vl53lmz_results_extract_block(&Dev, VL53LMZ_CNH_DATA_IDX,
                                             (uint8_t *)cnh_data_buffer,
                                             cnh_data_size);
      if (status != VL53LMZ_STATUS_OK) {
        printf("Error: vl53lmz_results_extract_block failed: %d\n", status);
        continue;
      }

      printf("Data Count : %3u\n", Dev.streamcount);
      for (agg_id = 0; agg_id < cnh_config.nb_of_aggregates; agg_id++) {
        vl53lmz_cnh_get_block_addresses(&cnh_config, agg_id, cnh_data_buffer,
                                        &p_hist, &p_hist_scaler, &p_ambient,
                                        &p_ambient_scaler);

        amb_value = ((float)*p_ambient) / (2 << *p_ambient_scaler);
        printf("Agg, %2d, Ambient, %.1f, Bins, ", agg_id, amb_value);

        for (bin_num = 0; bin_num < cnh_config.feature_length; bin_num++) {
          bin_value = ((float)p_hist[bin_num]) / (2 << p_hist_scaler[bin_num]);
          printf("%.1f, ", bin_value);
        }
        printf("\n");
      }
    }
  }

  return 0;
}

// Updated process_config function
uint8_t process_config(SensorConfig config) {
  uint8_t status, isAlive;

  /* Check if there is a VL53LMZ sensor connected */
  status = vl53lmz_is_alive(&Dev, &isAlive);
  if (!isAlive || status) {
    printf("VL53LMZ not detected at requested address\n");
    return status;
  }

  /* Initialise the VL53LMZ sensor */
  status = vl53lmz_init(&Dev);
  if (status) {
    printf("VL53LMZ ULD Loading failed\n");
    return status;
  }

  printf("Applying sensor configuration...\n");

  /*********************************/
  /*  Set basic ranging settings   */
  /*********************************/
  status = vl53lmz_set_resolution(&Dev, config.resolution);
  if (status != VL53LMZ_STATUS_OK) {
    printf("ERROR at %s(%d) : vl53lmz_set_resolution failed : %d\n", __func__,
           __LINE__, status);
    return status;
  }
  status = vl53lmz_set_ranging_mode(&Dev, config.ranging_mode);
  if (status != VL53LMZ_STATUS_OK) {
    printf("ERROR at %s(%d) : vl53lmz_set_ranging_mode failed : %d\n", __func__,
           __LINE__, status);
    return status;
  }
  status = vl53lmz_set_ranging_frequency_hz(&Dev, config.ranging_frequency_hz);
  if (status != VL53LMZ_STATUS_OK) {
    printf("ERROR at %s(%d) : vl53lmz_set_ranging_frequency_hz failed : %d\n",
           __func__, __LINE__, status);
    return status;
  }
  status = vl53lmz_set_integration_time_ms(&Dev, config.integration_time_ms);
  if (status != VL53LMZ_STATUS_OK) {
    printf("ERROR at %s(%d) : vl53lmz_set_integration_time_ms failed : %d\n",
           __func__, __LINE__, status);
    return status;
  }

  /*********************************/
  /*  CNH specific configuration   */
  /*********************************/
  // Initialize CNH configuration
  status |= vl53lmz_cnh_init_config(&cnh_config,          //
                                    config.cnh_start_bin, //
                                    config.cnh_num_bins,  //
                                    config.cnh_subsample);

  if (status != VL53LMZ_STATUS_OK) {
    printf("ERROR at %s(%d) : vl53lmz_cnh_init_config failed : %d\n", __func__,
           __LINE__, status);
    return status;
  }

  // Create aggregate map
  printf("Resolution: %d\n", config.resolution);
  status |= vl53lmz_cnh_create_agg_map(&cnh_config,        //
                                       config.resolution,  //
                                       config.agg_start_x, //
                                       config.agg_start_y, //
                                       config.agg_merge_x, //
                                       config.agg_merge_y, //
                                       config.agg_cols,    //
                                       config.agg_rows);

  if (status != VL53LMZ_STATUS_OK) {
    printf("ERROR at %s(%d) : vl53lmz_cnh_create_agg_map failed : %d\n",
           __func__, __LINE__, status);
    return status;
  }

  // Check that the requested configuration will not generate CNH data that
  // is too large for the available space on the sensor. Store the size of
  // data generate so we can next setup an optimize data transfer from
  // sensor to host.
  status = vl53lmz_cnh_calc_required_memory(&cnh_config, &cnh_data_size);
  if (status != VL53LMZ_STATUS_OK) {
    printf("ERROR at %s(%d) : vl53lmz_cnh_calc_required_memory : %d\n",
           __func__, __LINE__, status);
    if (cnh_data_size < 0) {
      printf("Required memory is too high : %lu.	Maximum is %lu!\n",
             cnh_data_size, VL53LMZ_CNH_MAX_DATA_BYTES);
    }
    return status;
  }

  // Send CNH configuration to the sensor
  status = vl53lmz_cnh_send_config(&Dev, &cnh_config);
  if (status != VL53LMZ_STATUS_OK) {
    printf("ERROR at %s(%d) : vl53lmz_cnh_send_config failed : %d\n", __func__,
           __LINE__, status);
    return status;
  }

  // Because we want to use a non-standard data transfer from the device we
  // can not use the standard vl53lmz_start_ranging() function, instead we
  // need to use vl53lmz_create_output_config() followed by
  // vl53lmz_send_output_config_and_start() This allows us to modify the
  // data transfer requested between the two functions.

  // Prepare output configuration
  status = vl53lmz_create_output_config(&Dev);
  if (status != VL53LMZ_STATUS_OK) {
    printf("Error: Creating output config failed: %u\n", status);
    return status;
  }

  // Add CNH data block
  union Block_header cnh_data_bh;
  cnh_data_bh.idx = VL53LMZ_CNH_DATA_IDX;
  cnh_data_bh.type = 4;
  cnh_data_bh.size = cnh_data_size / 4;
  status = vl53lmz_add_output_block(&Dev, cnh_data_bh.bytes);
  if (status != VL53LMZ_STATUS_OK) {
    printf("ERROR at %s(%d) : vl53lmz_add_output_block failed : %d\n", __func__,
           __LINE__, status);
    return status;
  }

  // Start ranging
  status = vl53lmz_send_output_config_and_start(&Dev);
  if (status != VL53LMZ_STATUS_OK) {
    printf("Error: Starting ranging failed: %u\n", status);
    return status;
  }

  HAL_Delay(1000);

  printf("Sensor configuration applied and ranging started.\n");

  return status;
}
