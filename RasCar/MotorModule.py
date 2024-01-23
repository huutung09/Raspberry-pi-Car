import RPi.GPIO as GPIO
from time import sleep
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
 
class Motor():
    def __init__(self, In2, In3, In4):
        
        self.In2 = In2 ## ~
        self.In3 = In3 ## ~
        self.In4 = In4 
        
        
        GPIO.setup(self.In2,GPIO.OUT)
        GPIO.setup(self.In3,GPIO.OUT)
        GPIO.setup(self.In4,GPIO.OUT)
        
        self.pwmA = GPIO.PWM(self.In2, 50)
        self.motorA_start()
        
        self.pwmB = GPIO.PWM(self.In3, 100)
        
    
    def stop_all(self,t=0):  
        #self.pwmA.stop()
        
        self.pwmB.stop()
        GPIO.output(self.In4, GPIO.LOW)
        
    def motorA_start(self):
        self.pwmA.start(0)
        duty = 2 + (76 / 18)
        self.pwmA.ChangeDutyCycle(duty)
        
    def motorA_forward(self, t=0):
        duty = 2 + (76 / 18)
        self.pwmA.ChangeDutyCycle(duty);
        sleep(t)
    
    def motorA_left(self, t=0):
        duty = 2 + (76-40) / 18
        self.pwmA.ChangeDutyCycle(duty);
        sleep(t)
        
    def motorA_right(self, t=0):
        duty = 2 + (76+40) / 18
        self.pwmA.ChangeDutyCycle(duty);
        sleep(t)
        
        
    def motorB_forward(self, speed, t=0):
        GPIO.output(self.In4, GPIO.HIGH)
        self.pwmB.ChangeDutyCycle(100 - speed)
        
    def motorB_back(self, speed, t=0):
        GPIO.output(self.In4, GPIO.LOW)
        self.pwmB.ChangeDutyCycle(100 - speed)
    
    def motorB_stop(self):
        GPIO.output(self.In4, GPIO.LOW)
        self.pwmB.stop()
        
    def motorB_start(self):
        self.pwmB.start(35);
    
    def gpip_cleanup(self):
        GPIO.cleanup()
        
 
def main():
    motor.motorB_forward(50)
    motor.motorA_right()
    sleep(2)
    motor.motorA_left()
    sleep(2)
    motor.motorA_forward()
    sleep(2)
    motor.motorB_start()
    motor.motorB_back(50)
    sleep(2)
    motor.motorB_stop()
    sleep(5)
    motor.motorB_forward(50)
    sleep(5)
    motor.motorB_start()
    motor.motorB_forward(50)
    sleep(5)
    motor.stop_all()
    motor.gpip_cleanup()
    print("goodbye")

    

 
if __name__ == '__main__':
    motor= Motor(13, 12, 16)
    motor.motorA_forward()
    sleep(2)
    motor.stop_all()
    motor.gpip_cleanup()
    print("goodbye")
    #maint()