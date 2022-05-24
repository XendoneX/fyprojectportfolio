import clr
file = 'OpenHardwareMonitorLib'
clr.AddReference(file)

from OpenHardwareMonitor import Hardware #Imports Hardware function from OpenHardwareMonitorLib.dll

hwtypes = ['Mainboard','SuperIO','CPU','RAM','GpuNvidia','GpuAti','TBalancer','Heatmaster','HDD'] #List of all hardware component variables in OpenHardwareMonitorLib.dll
sensortypes = ['Voltage','Clock','Temperature','Load','Fan','Flow','Control','Level','Factor','Power','Data','SmallData'] #List of all sensor value types in OpenHardwareMonitorLib.dll

def openhardwaremonitor(): #OpenHardwareMonitor opens a port that listens to hardware sensors
    handle = Hardware.Computer()
    handle.CPUEnabled = True
    handle.RAMEnabled = True
    handle.HDDEnabled = True
    handle.Open()
    return handle

def fetchCpuLoad(HardwareHandle): #Gets CpuLoad values
    for i in HardwareHandle.Hardware:  #Checks for every available computer hardware sensor
        i.Update() #Updates the list of available hardware sensors
        for sensor in i.Sensors: #Checks every sensor
            if sensor.SensorType == sensortypes.index('Load') and sensor.Name == 'CPU Total': #Looks for sensor with thec value type Load and sensor with the name 'CPU Total'
                return sensor.Value #Return the value

def fetchCpuPower(HardwareHandle):
    for i in HardwareHandle.Hardware:
        i.Update()
        for sensor in i.Sensors:
            if sensor.SensorType == sensortypes.index('Power') and sensor.Name == 'CPU Package':
                return sensor.Value

def fetchCpuTemp(HardwareHandle):
    for i in HardwareHandle.Hardware:
        i.Update()
        for sensor in i.Sensors:
            if sensor.SensorType == sensortypes.index('Temperature') and sensor.Name == 'CPU Package':
                return sensor.Value

def fetchHddTemp(HardwareHandle):
    for i in HardwareHandle.Hardware:
        i.Update()
        for sensor in i.Sensors:
            if sensor.SensorType == sensortypes.index('Temperature') and hwtypes[sensor.Hardware.HardwareType] == 'HDD':
                return sensor.Value

def fetchRamLoad(HardwareHandle):
    for i in HardwareHandle.Hardware:
        i.Update()
        for sensor in i.Sensors:
            if sensor.SensorType == sensortypes.index('Load') and hwtypes[sensor.Hardware.HardwareType] == 'RAM':
                return sensor.Value

def fetchRamUsage(HardwareHandle):
    for i in HardwareHandle.Hardware:
        i.Update()
        for sensor in i.Sensors:
            if sensor.SensorType == sensortypes.index('Data') and sensor.Name == 'Used Memory':
                return sensor.Value

if __name__ == "__main__": #Bug Testing
#while True:
    HardwareHandle = openhardwaremonitor()
    print("cpuLoad: ", fetch_cpuLoad(HardwareHandle))

    print("cpuPower: ", fetch_cpuPower(HardwareHandle))

    print("cpuTemp: ", fetch_cpuTemp(HardwareHandle))

    print("hddTemp: ", fetch_hddTemp(HardwareHandle))

    print("ramLoad: ", fetch_ramLoad(HardwareHandle))

    print("ramUsage: ", fetch_ramUsage(HardwareHandle))