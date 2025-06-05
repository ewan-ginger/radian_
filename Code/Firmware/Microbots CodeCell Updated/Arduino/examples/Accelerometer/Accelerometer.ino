#include <CodeCell.h>  // Make sure you've installed the CodeCell library

CodeCell myCodeCell;

void setup() {
  Serial.begin(115200);
  delay(1000);  // Wait for Serial to initialize
  myCodeCell.Init(MOTION_ROTATION + MOTION_LINEAR_ACC + MOTION_GYRO + MOTION_MAGNETOMETER);
}

void loop() {
  // Run sensor reading at 50 Hz
  if (myCodeCell.Run(80)) {
    float x, y, z;
    myCodeCell.Motion_RotationRead(x, y, z);

    // Print in a format suitable for Serial Plotter
    Serial.print("X:"); Serial.print(x);
    Serial.print(" Y:"); Serial.print(y);
    Serial.print(" Z:"); Serial.println(z);
  }
}
