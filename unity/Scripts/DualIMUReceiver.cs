/*
  DualIMUReceiver.cs
  -------------------
  Reads JSON lines from the ESP32 (firmware/dual_mpu6050_stream.ino) over
  a serial port and applies the two quaternions to two Transforms,
  simulating a two-segment joint (e.g. forearm -> hand).

  Setup:
    1. Attach this script to any GameObject in the scene (e.g. an empty
       "IMUManager" object).
    2. Drag your "Forearm" Transform into the Segment A slot.
    3. Drag your "Hand" Transform into the Segment B slot.
    4. Set the Port Name to match Tools > Port in Arduino IDE (e.g. "COM3").
    5. Press Play.

  Requires: Project Settings > Player > Api Compatibility Level = .NET Framework
*/

using System;
using System.IO.Ports;
using UnityEngine;

public class DualIMUReceiver : MonoBehaviour
{
    [Header("Serial Settings")]
    public string portName = "COM3";
    public int baudRate = 115200;

    [Header("Joint Transforms")]
    public Transform segmentA; // e.g. Forearm
    public Transform segmentB; // e.g. Hand

    private SerialPort serialPort;
    private Quaternion latestA = Quaternion.identity;
    private Quaternion latestB = Quaternion.identity;
    private bool hasNewData = false;

    void Start()
    {
        try
        {
            serialPort = new SerialPort(portName, baudRate);
            serialPort.ReadTimeout = 50;
            serialPort.Open();
            Debug.Log("Serial port opened: " + portName);
        }
        catch (Exception e)
        {
            Debug.LogError("Failed to open serial port: " + e.Message);
        }
    }

    void Update()
    {
        if (serialPort == null || !serialPort.IsOpen) return;

        try
        {
            while (serialPort.BytesToRead > 0)
            {
                string line = serialPort.ReadLine();
                ParseLine(line);
            }
        }
        catch (TimeoutException)
        {
            // Normal when no new data is ready yet, safe to ignore.
        }

        if (hasNewData)
        {
            if (segmentA != null) segmentA.localRotation = latestA;
            if (segmentB != null) segmentB.localRotation = latestB;
            hasNewData = false;
        }
    }

    // Minimal manual JSON parsing (no dependency needed) for the fixed
    // format: {"a":{"w":..,"x":..,"y":..,"z":..},"b":{"w":..,"x":..,"y":..,"z":..}}
    void ParseLine(string line)
    {
        try
        {
            float aw = ExtractFloat(line, "\"a\":{\"w\":");
            float ax = ExtractFloat(line, "\"x\":", line.IndexOf("\"a\":"));
            float ay = ExtractFloat(line, "\"y\":", line.IndexOf("\"a\":"));
            float az = ExtractFloat(line, "\"z\":", line.IndexOf("\"a\":"));

            int bIndex = line.IndexOf("\"b\":");
            float bw = ExtractFloat(line, "\"w\":", bIndex);
            float bx = ExtractFloat(line, "\"x\":", bIndex);
            float by = ExtractFloat(line, "\"y\":", bIndex);
            float bz = ExtractFloat(line, "\"z\":", bIndex);

            // MPU6050 DMP quaternion order is (w, x, y, z);
            // Unity's Quaternion constructor takes (x, y, z, w).
            latestA = new Quaternion(ax, ay, az, aw);
            latestB = new Quaternion(bx, by, bz, bw);
            hasNewData = true;
        }
        catch
        {
            // Incomplete/corrupted line, just skip this frame's data.
        }
    }

    float ExtractFloat(string source, string key, int startIndex = 0)
    {
        int keyIndex = source.IndexOf(key, startIndex);
        int valueStart = keyIndex + key.Length;
        int valueEnd = source.IndexOfAny(new char[] { ',', '}' }, valueStart);
        string valueStr = source.Substring(valueStart, valueEnd - valueStart);
        return float.Parse(valueStr, System.Globalization.CultureInfo.InvariantCulture);
    }

    void OnApplicationQuit()
    {
        if (serialPort != null && serialPort.IsOpen)
        {
            serialPort.Close();
        }
    }
}
