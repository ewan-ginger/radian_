#pragma once

#include <lvgl.h>
#include <Arduino.h>
#include "lv_conf.h"
#include "LVGL_Driver.h"
#include "PWR_Key.h"
#include "LVGL_Example.h"
#include "LVGL_Music.h"

#define BOOT_KEY_PIN     0


struct Simulated_XY{
  uint8_t points;     // Number of touch points
  uint16_t x;         /*!< X coordinate */
  uint16_t y;         /*!< Y coordinate */
};

extern lv_obj_t *current_obj; 
extern struct Simulated_XY touch_data;
void Simulated_Touch_Init();
void Simulated_Touch(void);