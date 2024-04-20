#include <BleMouse.h>

BleMouse bleMouse;
String input = "";
int FIELD_WIDTH = 492;
int MINIMUM_TICK = 3;

void setup() {
  Serial.begin(115200);
  Serial.println("Starting");
  bleMouse.begin();
  delay(1000);
  move(-1000);
  move(165 + FIELD_WIDTH/2);
}

void loop() {
  input = Serial.readString();
  if(input != ""){
    float x = atof(input.c_str());
    int pos = round(FIELD_WIDTH / 2.0 * (x + 1.0)) - FIELD_WIDTH / 2;
    click(pos);
  }
}

void click(int pos) {
  move(pos);
  bleMouse.click();
  Serial.println("Clicked");
  delay(20);
  move(-pos);
}

void move(int v) {
  v = floor(v / MINIMUM_TICK);
  int d = 0;
  if(v >= 0){
    d = 1;
  }else{
    d = -1;
  }
  for(int i=0;i<abs(v);i++){
    bleMouse.move(d * MINIMUM_TICK,0,0);
    delay(10);
  }
}
