################################################################################

-include ../makefile.init

RM := rm -rf

# All of the sources participating in the build are defined here
-include sources.mk
-include VL53LMZ_ULD_API/src/subdir.mk
-include Platform/subdir.mk
-include Drivers/STM32F4xx_HAL_Driver/Src/subdir.mk
-include Core/Startup/subdir.mk
-include Core/Src/subdir.mk
-include objects.mk

ifneq ($(MAKECMDGOALS),clean)
ifneq ($(strip $(S_DEPS)),)
-include $(S_DEPS)
endif
ifneq ($(strip $(S_UPPER_DEPS)),)
-include $(S_UPPER_DEPS)
endif
ifneq ($(strip $(C_DEPS)),)
-include $(C_DEPS)
endif
endif

-include ../makefile.defs

OPTIONAL_TOOL_DEPS := \
$(wildcard ../makefile.defs) \
$(wildcard ../makefile.init) \
$(wildcard ../makefile.targets) \

BUILD_ARTIFACT_NAME := cc_hardware_vl53l8ch_driver
BUILD_ARTIFACT_EXTENSION := elf
BUILD_ARTIFACT_PREFIX :=
BUILD_ARTIFACT := $(BUILD_ARTIFACT_PREFIX)$(BUILD_ARTIFACT_NAME)$(if $(BUILD_ARTIFACT_EXTENSION),.$(BUILD_ARTIFACT_EXTENSION),)

# Add inputs and outputs from these tool invocations to the build variables
EXECUTABLES += \
cc_hardware_vl53l8ch_driver.elf \

MAP_FILES += \
cc_hardware_vl53l8ch_driver.map \

SIZE_OUTPUT += \
default.size.stdout \

OBJDUMP_LIST += \
cc_hardware_vl53l8ch_driver.list \

OBJCOPY_BIN += \
cc_hardware_vl53l8ch_driver.bin \

# All Target
all: main-build

# Main-build Target
main-build: cc_hardware_vl53l8ch_driver.elf secondary-outputs

# Tool invocations
cc_hardware_vl53l8ch_driver.elf cc_hardware_vl53l8ch_driver.map: $(OBJS) $(USER_OBJS) ../STM32F401RETX_FLASH.ld makefile objects.list $(OPTIONAL_TOOL_DEPS)
	arm-none-eabi-gcc -o "cc_hardware_vl53l8ch_driver.elf" @"objects.list" $(USER_OBJS) $(LIBS) -mcpu=cortex-m4 -T"../STM32F401RETX_FLASH.ld" --specs=nosys.specs -Wl,-Map="cc_hardware_vl53l8ch_driver.map" -Wl,--gc-sections -static --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -u _printf_float -Wl,--start-group -lc -lm -Wl,--end-group
	@echo 'Finished building target: $@'
	@echo ' '

default.size.stdout: $(EXECUTABLES) makefile objects.list $(OPTIONAL_TOOL_DEPS)
	arm-none-eabi-size  $(EXECUTABLES)
	@echo 'Finished building: $@'
	@echo ' '

cc_hardware_vl53l8ch_driver.list: $(EXECUTABLES) makefile objects.list $(OPTIONAL_TOOL_DEPS)
	arm-none-eabi-objdump -h -S $(EXECUTABLES) > "cc_hardware_vl53l8ch_driver.list"
	@echo 'Finished building: $@'
	@echo ' '

cc_hardware_vl53l8ch_driver.bin: $(EXECUTABLES) makefile objects.list $(OPTIONAL_TOOL_DEPS)
	arm-none-eabi-objcopy  -O binary $(EXECUTABLES) "cc_hardware_vl53l8ch_driver.bin"
	@echo 'Finished building: $@'
	@echo ' '

# Upload Target
upload: cc_hardware_vl53l8ch_driver.bin
	@echo 'Uploading cc_hardware_vl53l8ch_driver.bin to $(PORT)...'
	@if [ -z "$(PORT)" ]; then \
		echo "Error: PORT variable not set. Usage: make upload PORT=<PORT>"; \
		exit 1; \
	fi
	# Replace the command below with the appropriate upload command for your hardware
	cp cc_hardware_vl53l8ch_driver.bin $(PORT)

# Other Targets
clean:
	-$(RM) cc_hardware_vl53l8ch_driver.bin cc_hardware_vl53l8ch_driver.elf cc_hardware_vl53l8ch_driver.list cc_hardware_vl53l8ch_driver.map default.size.stdout
	-@echo ' '

secondary-outputs: $(SIZE_OUTPUT) $(OBJDUMP_LIST) $(OBJCOPY_BIN)

fail-specified-linker-script-missing:
	@echo 'Error: Cannot find the specified linker script. Check the linker settings in the build configuration.'
	@exit 2

warn-no-linker-script-specified:
	@echo 'Warning: No linker script specified. Check the linker settings in the build configuration.'

.PHONY: all clean dependents main-build fail-specified-linker-script-missing warn-no-linker-script-specified upload

-include ../makefile.targets
