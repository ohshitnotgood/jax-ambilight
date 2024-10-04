#include <Arduino.h>
#include <FastLED.h>

using namespace std;

#define NUM_LEDS 120
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
void turn_off_leds();

String old_frame = "";

int tm_ctr = 0;

int c = 0;

void setup()
{
    FastLED.addLeds<WS2812, DATA_PIN>(leds, NUM_LEDS);
    turn_off_leds();
    Serial.begin(BAUD_RATE);
}

void loop()
{
    if (Serial.available())
    {
        read_next_frame();
        update_led_strip_zones();
        show_leds();
        acknowledge_success();
        // Serial.readStringUntil(';');
        // leds[c] = CRGB(255, 0, 0);
        // if (c > 0) {
        //     leds[c - 1] = CRGB(0, 0, 0);
        // }
        // show_leds();
        // c++;
        // if (c >= 120) {
        //     c = 0;
        // }
    }
}

void acknowledge_success()
{
    Serial.write("ack");
}

void show_leds()
{
    FastLED.show();
}

void read_next_frame()
{
    String next_frame = Serial.readStringUntil(';');
    
    // if (next_frame == old_frame) {
    //     return;
    // } else {
    //     old_frame = next_frame;
    // }

    zone_thirteen = CRGB((int)next_frame.charAt(1), (int)next_frame.charAt(0), (int)next_frame.charAt(2));
    next_frame.remove(0, 3);

    zone_twelve = CRGB((int)next_frame.charAt(1), (int)next_frame.charAt(0), (int)next_frame.charAt(2));
    next_frame.remove(0, 3);

    zone_eleven = CRGB((int)next_frame.charAt(1), (int)next_frame.charAt(0), (int)next_frame.charAt(2));
    next_frame.remove(0, 3);

    zone_three = CRGB((int)next_frame.charAt(1), (int)next_frame.charAt(0), (int)next_frame.charAt(2));
    next_frame.remove(0, 3);

    zone_two = CRGB((int)next_frame.charAt(1), (int)next_frame.charAt(0), (int)next_frame.charAt(2));
    next_frame.remove(0, 3);

    zone_one = CRGB((int)next_frame.charAt(1), (int)next_frame.charAt(0), (int)next_frame.charAt(2));
    next_frame.remove(0, 3);

    zone_zero = CRGB((int)next_frame.charAt(1), (int)next_frame.charAt(0), (int)next_frame.charAt(2));
    next_frame.remove(0, 3);

    zone_eight = CRGB((int)next_frame.charAt(1), (int)next_frame.charAt(0), (int)next_frame.charAt(2));
    next_frame.remove(0, 3);

    zone_nine = CRGB((int)next_frame.charAt(1), (int)next_frame.charAt(0), (int)next_frame.charAt(2));
    next_frame.remove(0, 3);

    zone_ten = CRGB((int)next_frame.charAt(1), (int)next_frame.charAt(0), (int)next_frame.charAt(2));
    next_frame.remove(0, 3);

    zone_four = CRGB((int)next_frame.charAt(1), (int)next_frame.charAt(0), (int)next_frame.charAt(2));
    next_frame.remove(0, 3);

    zone_five = CRGB((int)next_frame.charAt(1), (int)next_frame.charAt(0), (int)next_frame.charAt(2));
    next_frame.remove(0, 3);

    zone_six = CRGB((int)next_frame.charAt(1), (int)next_frame.charAt(0), (int)next_frame.charAt(2));
    next_frame.remove(0, 3);

    zone_seven = CRGB((int)next_frame.charAt(1), (int)next_frame.charAt(0), (int)next_frame.charAt(2));
    next_frame.remove(0, 3);
}

void update_led_strip_zones()
{
    // Update bottom thirteen
    leds[0] = zone_zero;
    leds[1] = zone_zero;
    leds[2] = zone_zero;
    
    leds[7] = zone_one;
    leds[8] = zone_one;
    leds[9] = zone_one;

    
    leds[12] = zone_two;
    leds[13] = zone_two;
    leds[14] = zone_two;       
    
    leds[20] = zone_three;
    leds[21] = zone_three;
    leds[22] = zone_three;
    

    leds[23] = zone_four; 
    leds[24] = zone_four; 
    leds[25] = zone_four; 
    
    leds[30] = zone_five;
    leds[31] = zone_five;
    leds[32] = zone_five;

    leds[40] = zone_six;
    leds[41] = zone_six;
    leds[42] = zone_six;    
    
    leds[48] = zone_seven;
    leds[49] = zone_seven;
    leds[50] = zone_seven;
    
    leds[58] = zone_eight;
    leds[59] = zone_eight;
    leds[60] = zone_eight;

    leds[68] = zone_nine;
    leds[69] = zone_nine;
    leds[70] = zone_nine;

    leds[76] = zone_ten;
    leds[77] = zone_ten;
    leds[78] = zone_ten;

    leds[88] = zone_eleven;
    leds[89] = zone_eleven;
    leds[90] = zone_eleven;
    
    leds[97] = zone_twelve;
    leds[98] = zone_twelve;
    leds[99] = zone_twelve;
    
    leds[108] = zone_thirteen;
    leds[109] = zone_thirteen;
    leds[110] = zone_thirteen;
}

void turn_off_leds()
{
    // Update bottom thirteen
    leds[0] = CRGB(0, 0, 0);
    leds[8] = CRGB(0, 0, 0);   // GRB
    leds[16] = CRGB(0, 0, 0);  // GRB
    leds[24] = CRGB(0, 0, 0);  // GRB
    leds[32] = CRGB(0, 0, 0);  // GRB
    leds[40] = CRGB(0, 0, 0);  // GRB
    leds[48] = CRGB(0, 0, 0);  // GRB
    leds[52] = CRGB(0, 0, 0);  // GRB
    leds[60] = CRGB(0, 0, 0);  // GRB
    leds[70] = CRGB(0, 0, 0);  // GRB
    leds[80] = CRGB(0, 0, 0);  // GRB
    leds[90] = CRGB(0, 0, 0);  // GRB
    leds[105] = CRGB(0, 0, 0); // GRB
    leds[110] = CRGB(0, 0, 0); // GRB

    show_leds();
}