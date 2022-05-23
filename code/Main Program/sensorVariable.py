import clr
file = 'OpenHardwareMonitorLib'
clr.AddReference(file)

from OpenHardwareMonitor import Hardware

hwtypes = ['Mainboard','SuperIO','CPU','RAM','GpuNvidia','GpuAti','TBalancer','Heatmaster','HDD']
sensortypes = ['Voltage','Clock','Temperature','Load','Fan','Flow','Control','Level','Factor','Power','Data','SmallData']

def openhardwaremonitor():
    handle = Hardware.Computer()
    handle.CPUEnabled = True
    handle.RAMEnabled = True
    handle.HDDEnabled = True
    handle.Open()
    return handle

def fetch_cpuLoad(HardwareHandle):
    for i in HardwareHandle.Hardware:
        i.Update()
        for sensor in i.Sensors:
            if sensor.SensorType == sensortypes.index('Load') and sensor.Name == 'CPU Total': #CPU Load
                return sensor.Value

def fetch_cpuPower(HardwareHandle):
    for i in HardwareHandle.Hardware:
        i.Update()
        for sensor in i.Sensors:
            if sensor.SensorType == sensortypes.index('Power') and sensor.Name == 'CPU Package':
                return sensor.Value

def fetch_cpuTemp(HardwareHandle):
    for i in HardwareHandle.Hardware:
        i.Update()
        for sensor in i.Sensors:
            if sensor.SensorType == sensortypes.index('Temperature') and sensor.Name == 'CPU Package':
                return sensor.Value

def fetch_hddTemp(HardwareHandle):
    for i in HardwareHandle.Hardware:
        i.Update()
        for sensor in i.Sensors:
            if sensor.SensorType == sensortypes.index('Temperature') and hwtypes[sensor.Hardware.HardwareType] == 'HDD':
                return sensor.Value

def fetch_ramLoad(HardwareHandle):
    for i in HardwareHandle.Hardware:
        i.Update()
        for sensor in i.Sensors:
            if sensor.SensorType == sensortypes.index('Load') and hwtypes[sensor.Hardware.HardwareType] == 'RAM':
                return sensor.Value

def fetch_ramUsage(HardwareHandle):
    for i in HardwareHandle.Hardware:
        i.Update()
        for sensor in i.Sensors:
            if sensor.SensorType == sensortypes.index('Data') and sensor.Name == 'Used Memory':
                return sensor.Value

if __name__ == "__main__":
#while True:
    HardwareHandle = openhardwaremonitor()
    print("cpuLoad: ", fetch_cpuLoad(HardwareHandle))

    print("cpuPower: ", fetch_cpuPower(HardwareHandle))

    print("cpuTemp: ", fetch_cpuTemp(HardwareHandle))

    print("hddTemp: ", fetch_hddTemp(HardwareHandle))

    print("ramLoad: ", fetch_ramLoad(HardwareHandle))

    print("ramUsage: ", fetch_ramUsage(HardwareHandle))