################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (12.3.rel1)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables
C_SRCS += \
../Platform/platform.c

OBJS += \
./Platform/platform.o

C_DEPS += \
./Platform/platform.d


# Each subdirectory must supply rules for building sources it contributes
Platform/%.o Platform/%.su Platform/%.cyclo: ../Platform/%.c Platform/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m4 -std=gnu11 -DUSE_HAL_DRIVER -DSTM32F401xE -c -I../Platform -I../VL53LMZ_ULD_API/inc -I../Core/Inc -I../Drivers/STM32F4xx_HAL_Driver/Inc -I../Drivers/STM32F4xx_HAL_Driver/Inc/Legacy -I../Drivers/CMSIS/Device/ST/STM32F4xx/Include -I../Drivers/CMSIS/Include -Os -ffunction-sections -fdata-sections -Wall -fstack-usage -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -o "$@"

clean: clean-Platform

clean-Platform:
	-$(RM) ./Platform/platform.cyclo ./Platform/platform.d ./Platform/platform.o ./Platform/platform.su

.PHONY: clean-Platform
