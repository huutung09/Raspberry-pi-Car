import RPi.GPIO as GPIO
from time import sleep
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
 
class Motor():
    def __init__(self, In1, In2, In3, In4):
        self.In1 = In1
        self.In2 = In2
        self.In3 = In3
        self.In4 = In4
        
        GPIO.setup(self.In1,GPIO.OUT)
        GPIO.setup(self.In2,GPIO.OUT)
        GPIO.setup(self.In3,GPIO.OUT)
        GPIO.setup(self.In4,GPIO.OUT)
        
        self.pwmB = GPIO.PWM(self.In3, 100);
        self.pwmB.start(0);
 
    def stop(self,t=0):
        GPIO.output(self.In1, GPIO.LOW)
        #self.pwmA.ChangeDutyCycle(0);
        self.pwmB.ChangeDutyCycle(0);
        GPIO.output(self.In4, GPIO.LOW)
        sleep(t)
        
    def motorA_forward(self, t=0):
        GPIO.output(self.In1, GPIO.LOW)
        GPIO.output(self.In2, GPIO.LOW)
    
    def motorA_left(self, t=0):
        GPIO.output(self.In1, GPIO.HIGH)
        GPIO.output(self.In2, GPIO.LOW)
        
    def motorA_right(self, t=0):
        GPIO.output(self.In1, GPIO.LOW)
        GPIO.output(self.In2, GPIO.HIGH)
        
        
        
    def motorB_forward(self, speed, t=0):
        GPIO.output(self.In4, GPIO.HIGH)
        self.pwmB.ChangeDutyCycle(100 - speed)
        
    def motorB_back(self, speed, t=0):
        GPIO.output(self.In4, GPIO.LOW)
        self.pwmB.ChangeDutyCycle(100 - speed)
        
        
 
def main():
    motor.motorB_forward(50)
    sleep(5)
    motor.motorB_back(50)
    sleep(5)
    motor.motorA_right()
    motor.motorA_left()
    sleep(2)
    motor.motorA_forward(50)
    sleep(2)
    motor.stop()
    GPIO.cleanup()

    

 
if __name__ == '__main__':
    motor= Motor(6, 13, 12, 16)
    main()
    
    
