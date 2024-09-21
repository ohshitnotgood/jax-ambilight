#include <Arduino.h>
#include <FastLED.h>

using namespace std;

#define NUM_LEDS 14
#define BAUD_RATE 9600
#define DATA_PIN 13
#define NUM_ZONES 14
#define NUM_LEDS_IN_ROW 30
#define NUM_LEDS_IN_COL 15

CRGB leds[NUM_LEDS];
CRGB zones[NUM_ZONES];

CRGB zone_zero;
CRGB zone_one;
CRGB zone_two;
CRGB zone_three;
CRGB zone_four;
CRGB zone_five;
CRGB zone_six;
CRGB zone_seven;
CRGB zone_eight;
CRGB zone_nine;
CRGB zone_ten;
CRGB zone_eleven;
CRGB zone_twelve;
CRGB zone_thirteen;

void read_next_frame();
void show_leds();
void update_leds();
void update_led_strip_zones();
void acknowledge_success();

void setup()
{
    FastLED.addLeds<WS2812, DATA_PIN>(leds, NUM_LEDS);
    leds[0] = CRGB(0, 255, 0);
    leds[1] = CRGB(255, 0, 0); // GRB
    leds[2] = CRGB(255, 0, 0); // GRB
    leds[3] = CRGB(255, 0, 0); // GRB
    leds[4] = CRGB(255, 0, 0); // GRB
    leds[5] = CRGB(255, 0, 0); // GRB
    leds[6] = CRGB(255, 0, 0); // GRB
    leds[7] = CRGB(255, 0, 0); // GRB
    leds[8] = CRGB(255, 0, 0); // GRB
    leds[9] = CRGB(255, 0, 0); // GRB
    leds[10] = CRGB(255, 0, 0); // GRB
    leds[11] = CRGB(255, 0, 0); // GRB
    leds[12] = CRGB(255, 0, 0); // GRB
    leds[13] = CRGB(255, 0, 0); // GRB

    FastLED.show();
    Serial.begin(BAUD_RATE);
}

void loop()
{
    if (Serial.available())
    {
        read_next_frame();
        update_leds();
        show_leds();
        acknowledge_success();
    }
}

void acknowledge_success()
{
    Serial.write("ack\n");
}

void update_leds() 
{
    leds[0] = zone_zero;
    leds[1] = zone_one;
    leds[2] = zone_two;
    leds[3] = zone_three;
    leds[4] = zone_four;
    leds[5] = zone_five;
    leds[6] = zone_six;
    leds[7] = zone_seven;
    leds[8] = zone_eight;
    leds[9] = zone_nine;
    leds[10] = zone_ten;
    leds[11] = zone_eleven;
    leds[12] = zone_twelve;
    leds[13] = zone_thirteen;
}

void show_leds() {
    FastLED.show();
}

void read_next_frame()
{
    if (Serial.available() > 0)
    {
        // Serial.println("Data is now available\n");
        // String next_frame = Serial.readString();
        String next_frame = Serial.readStringUntil(';');

        zone_zero = CRGB(next_frame.substring(3, 6).toInt(), next_frame.substring(0, 3).toInt(), next_frame.substring(6, 9).toInt());
        next_frame.remove(0, 9);

        zone_one = CRGB(next_frame.substring(3, 6).toInt(), next_frame.substring(0, 3).toInt(), next_frame.substring(6, 9).toInt());
        next_frame.remove(0, 9);
        
        zone_two = CRGB(next_frame.substring(3, 6).toInt(), next_frame.substring(0, 3).toInt(), next_frame.substring(6, 9).toInt());
        next_frame.remove(0, 9);

        zone_three = CRGB(next_frame.substring(3, 6).toInt(), next_frame.substring(0, 3).toInt(), next_frame.substring(6, 9).toInt());
        next_frame.remove(0, 9);
        
        zone_four = CRGB(next_frame.substring(3, 6).toInt(), next_frame.substring(0, 3).toInt(), next_frame.substring(6, 9).toInt());
        next_frame.remove(0, 9);

        zone_five = CRGB(next_frame.substring(3, 6).toInt(), next_frame.substring(0, 3).toInt(), next_frame.substring(6, 9).toInt());
        next_frame.remove(0, 9);

        zone_six = CRGB(next_frame.substring(3, 6).toInt(), next_frame.substring(0, 3).toInt(), next_frame.substring(6, 9).toInt());
        next_frame.remove(0, 9);

        zone_seven = CRGB(next_frame.substring(3, 6).toInt(), next_frame.substring(0, 3).toInt(), next_frame.substring(6, 9).toInt());
        next_frame.remove(0, 9);

        zone_eight = CRGB(next_frame.substring(3, 6).toInt(), next_frame.substring(0, 3).toInt(), next_frame.substring(6, 9).toInt());
        next_frame.remove(0, 9);

        zone_nine = CRGB(next_frame.substring(3, 6).toInt(), next_frame.substring(0, 3).toInt(), next_frame.substring(6, 9).toInt());
        next_frame.remove(0, 9);
        
        zone_ten = CRGB(next_frame.substring(3, 6).toInt(), next_frame.substring(0, 3).toInt(), next_frame.substring(6, 9).toInt());
        next_frame.remove(0, 9);
        
        zone_eleven = CRGB(next_frame.substring(3, 6).toInt(), next_frame.substring(0, 3).toInt(), next_frame.substring(6, 9).toInt());
        next_frame.remove(0, 9);
        
        zone_twelve = CRGB(next_frame.substring(3, 6).toInt(), next_frame.substring(0, 3).toInt(), next_frame.substring(6, 9).toInt());
        next_frame.remove(0, 9);
        
        zone_thirteen = CRGB(next_frame.substring(3, 6).toInt(), next_frame.substring(0, 3).toInt(), next_frame.substring(6, 9).toInt());
        next_frame.remove(0, 9);
        
        // zone_fourteen = CRGB(next_frame.substring(3, 6).toInt(), next_frame.substring(0, 3).toInt(), next_frame.substring(6, 9).toInt());
        // next_frame.remove(0, 9);
        
        // zone_fifteen = CRGB(next_frame.substring(3, 6).toInt(), next_frame.substring(0, 3).toInt(), next_frame.substring(6, 9).toInt());
        // next_frame.remove(0, 9);
        
        // zone_sixteen = CRGB(next_frame.substring(3, 6).toInt(), next_frame.substring(0, 3).toInt(), next_frame.substring(6, 9).toInt());
        // next_frame.remove(0, 9);
        
        // zone_seventeen = CRGB(next_frame.substring(3, 6).toInt(), next_frame.substring(0, 3).toInt(), next_frame.substring(6, 9).toInt());
        // next_frame.remove(0, 9);
        
        // zone_eighteen = CRGB(next_frame.substring(3, 6).toInt(), next_frame.substring(0, 3).toInt(), next_frame.substring(6, 9).toInt());
        // next_frame.remove(0, 9);
        
        // zone_nineteen = CRGB(next_frame.substring(3, 6).toInt(), next_frame.substring(0, 3).toInt(), next_frame.substring(6, 9).toInt());
        // next_frame.remove(0, 9);
        
        // zone_twenty = CRGB(next_frame.substring(3, 6).toInt(), next_frame.substring(0, 3).toInt(), next_frame.substring(6, 9).toInt());
        // next_frame.remove(0, 9);
        
        // zone_twentyone = CRGB(next_frame.substring(3, 6).toInt(), next_frame.substring(0, 3).toInt(), next_frame.substring(6, 9).toInt());
        // next_frame.remove(0, 9);
        
        // zone_twentytwo = CRGB(next_frame.substring(3, 6).toInt(), next_frame.substring(0, 3).toInt(), next_frame.substring(6, 9).toInt());
        // next_frame.remove(0, 9);
        
        // zone_twentythree = CRGB(next_frame.substring(3, 6).toInt(), next_frame.substring(0, 3).toInt(), next_frame.substring(6, 9).toInt());
        // next_frame.remove(0, 9);
        
    }
}

void update_led_strip_zones() 
{
    // Update bottom thirteen
    for (int i = 0; i < 8;) {
        leds[i] = zone_thirteen;
    }

    for (int i = 8; i < 14;) {
        leds[i] = zone_twelve;
    }

    for (int i = 14; i < 20;) {
        leds[i] = zone_eleven;
    }
    
    for (int i = 20; i < 28;) {
        leds[i] = zone_ten;
    }
    
    for (int i = 28; i < 32;) {
        leds[i] = zone_nine;
    }

    for (int i = 32; i < 36;) {
        leds[i] = zone_nine;
    }
    
    leds[0] = zone_thirteen;
    leds[1] = zone_thirteen;
    leds[2] = zone_thirteen;
    leds[3] = zone_thirteen;
    leds[4] = zone_thirteen;
    leds[5] = zone_thirteen;
    leds[6] = zone_thirteen;
    leds[7] = zone_thirteen;

    
}

// // 255
// // 255
// // 255
// // znc
// // 232
// // 33
// // 145
// // znc