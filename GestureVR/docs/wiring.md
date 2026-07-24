# Hardware Wiring — Dual MPU6050 + ESP32

## Components
- 1x ESP32 Dev Module
- 2x MPU6050 (GY-521 breakout)
- Breadboard
- Jumper wires

## Sensor roles
| Sensor | I2C Address | Placement | Role |
|---|---|---|---|
| A (mpuA) | 0x68 | Forearm | Reference / parent segment |
| B (mpuB) | 0x69 | Hand / wrist | Child segment — combined with A gives joint rotation |

## Power rails
1. ESP32 3.3V -> breadboard "+" rail
2. ESP32 GND -> breadboard "-" rail

## Sensor A (address 0x68 — default, AD0 unconnected)
| Pin | Connects to |
|---|---|
| VCC | Breadboard "+" rail |
| GND | Breadboard "-" rail |
| SDA | ESP32 GPIO21 |
| SCL | ESP32 GPIO22 |
| AD0 | Not connected (defaults LOW -> 0x68) |

## Sensor B (address 0x69 — AD0 tied high)
| Pin | Connects to |
|---|---|
| VCC | Breadboard "+" rail (same rail as Sensor A) |
| GND | Breadboard "-" rail (same rail as Sensor A) |
| SDA | ESP32 GPIO21 (shared bus with Sensor A) |
| SCL | ESP32 GPIO22 (shared bus with Sensor A) |
| AD0 | Same row as Sensor B's own VCC pin (ties AD0 to 3.3V -> 0x69) |

## Verifying the wiring
Before flashing any DMP firmware, run `firmware/i2c_scanner/i2c_scanner.ino`
and confirm the Serial Monitor reports **both** `0x68` and `0x69`. If only
one address appears, see the troubleshooting notes below.

## Common issues encountered while building this
- **Full-size breadboards often have a gap splitting the power rail in
  half.** A wire plugged into one half won't be electrically connected
  to the other half. If GND/VCC seem intermittent, check for this gap.
- **AD0 must be actively wired to 3.3V for Sensor B** — leaving it
  "unconnected" is not the same as tying it LOW; it can float and
  cause unreliable address behavior. Always wire it explicitly.
- **A loose GND connection causes an LED to flicker/flash rather than
  stay lit**, and can make address detection look inconsistent even
  when AD0 itself is wired correctly — always verify GND stability
  first if a sensor's I2C address seems to misbehave.
