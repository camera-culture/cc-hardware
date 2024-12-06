#ifndef __APP_H
#define __APP_H

#ifdef __cplusplus
extern "C" {
#endif

#include "stm32f4xx_hal.h"

int app(UART_HandleTypeDef *huart2);

#ifdef __cplusplus
}
#endif

#endif
