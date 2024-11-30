################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (12.3.rel1)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables
C_SRCS += \
../VL53LMZ_ULD_API/src/vl53lmz_api.c \
../VL53LMZ_ULD_API/src/vl53lmz_plugin_cnh.c \
../VL53LMZ_ULD_API/src/vl53lmz_plugin_detection_thresholds.c \
../VL53LMZ_ULD_API/src/vl53lmz_plugin_motion_indicator.c \
../VL53LMZ_ULD_API/src/vl53lmz_plugin_xtalk.c

OBJS += \
./VL53LMZ_ULD_API/src/vl53lmz_api.o \
./VL53LMZ_ULD_API/src/vl53lmz_plugin_cnh.o \
./VL53LMZ_ULD_API/src/vl53lmz_plugin_detection_thresholds.o \
./VL53LMZ_ULD_API/src/vl53lmz_plugin_motion_indicator.o \
./VL53LMZ_ULD_API/src/vl53lmz_plugin_xtalk.o

C_DEPS += \
./VL53LMZ_ULD_API/src/vl53lmz_api.d \
./VL53LMZ_ULD_API/src/vl53lmz_plugin_cnh.d \
./VL53LMZ_ULD_API/src/vl53lmz_plugin_detection_thresholds.d \
./VL53LMZ_ULD_API/src/vl53lmz_plugin_motion_indicator.d \
./VL53LMZ_ULD_API/src/vl53lmz_plugin_xtalk.d


# Each subdirectory must supply rules for building sources it contributes
VL53LMZ_ULD_API/src/%.o VL53LMZ_ULD_API/src/%.su VL53LMZ_ULD_API/src/%.cyclo: ../VL53LMZ_ULD_API/src/%.c VL53LMZ_ULD_API/src/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m4 -std=gnu11 -DUSE_HAL_DRIVER -DSTM32F401xE -c -I../Platform -I../VL53LMZ_ULD_API/inc -I../Core/Inc -I../Drivers/STM32F4xx_HAL_Driver/Inc -I../Drivers/STM32F4xx_HAL_Driver/Inc/Legacy -I../Drivers/CMSIS/Device/ST/STM32F4xx/Include -I../Drivers/CMSIS/Include -Os -ffunction-sections -fdata-sections -Wall -fstack-usage -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -o "$@"

clean: clean-VL53LMZ_ULD_API-2f-src

clean-VL53LMZ_ULD_API-2f-src:
	-$(RM) ./VL53LMZ_ULD_API/src/vl53lmz_api.cyclo ./VL53LMZ_ULD_API/src/vl53lmz_api.d ./VL53LMZ_ULD_API/src/vl53lmz_api.o ./VL53LMZ_ULD_API/src/vl53lmz_api.su ./VL53LMZ_ULD_API/src/vl53lmz_plugin_cnh.cyclo ./VL53LMZ_ULD_API/src/vl53lmz_plugin_cnh.d ./VL53LMZ_ULD_API/src/vl53lmz_plugin_cnh.o ./VL53LMZ_ULD_API/src/vl53lmz_plugin_cnh.su ./VL53LMZ_ULD_API/src/vl53lmz_plugin_detection_thresholds.cyclo ./VL53LMZ_ULD_API/src/vl53lmz_plugin_detection_thresholds.d ./VL53LMZ_ULD_API/src/vl53lmz_plugin_detection_thresholds.o ./VL53LMZ_ULD_API/src/vl53lmz_plugin_detection_thresholds.su ./VL53LMZ_ULD_API/src/vl53lmz_plugin_motion_indicator.cyclo ./VL53LMZ_ULD_API/src/vl53lmz_plugin_motion_indicator.d ./VL53LMZ_ULD_API/src/vl53lmz_plugin_motion_indicator.o ./VL53LMZ_ULD_API/src/vl53lmz_plugin_motion_indicator.su ./VL53LMZ_ULD_API/src/vl53lmz_plugin_xtalk.cyclo ./VL53LMZ_ULD_API/src/vl53lmz_plugin_xtalk.d ./VL53LMZ_ULD_API/src/vl53lmz_plugin_xtalk.o ./VL53LMZ_ULD_API/src/vl53lmz_plugin_xtalk.su

.PHONY: clean-VL53LMZ_ULD_API-2f-src
