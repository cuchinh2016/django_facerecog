import os
import django
import cv2
from django.utils import timezone
from datetime import timedelta

# Set up Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'face_recog_app.settings')
django.setup()
from main_services.models import History

# # Now you can use Django models to interact with the database
# def update_database(value):
#     # Example: Create a new History entry
#     new_entry = History(name_people=value)  # Adjust fields as needed
#     new_entry.save()
#     print("Database updated with new entry.")
def update_database(value):
    # Define the time window
    now = timezone.now()
    thirty_minutes_ago = now - timedelta(minutes=30)

    # Check for existing entries within the last 30 minutes
    existing_entries = History.objects.filter(
        name_people=value,
        datetime_appear__gte=thirty_minutes_ago
    )

    if existing_entries.exists():
        print("An entry with this name has been added within the last 30 minutes.")
    else:
        # Create and save the new entry
        new_entry = History(name_people=value)
        new_entry.save()
        print("Database updated with new entry.")

if __name__ == "__main__":
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    # print(os.path.join(os.getcwd(), 'main_services', 'Inference', 'trainer', 'trainer.yml'))
    recognizer.read(os.path.join(os.getcwd(), 'trainer', 'trainer.yml'))
    # print(os.path.join(os.getcwd(), 'main_services', 'Inference', 'haarcascade_frontalface_default.xml'))
    cascadePath = os.path.join(os.getcwd(), 'haarcascade_frontalface_default.xml')
    faceCascade = cv2.CascadeClassifier(cascadePath)

    font = cv2.FONT_HERSHEY_SIMPLEX

    #iniciate id counter
    id = 0

    # names related to ids: example ==> Marcelo: id=1,  etc
    names = ['None', 'Khac Chinh', 'Messi','', 'Khac Chinh 2']

    # Initialize and start realtime video capture
    cam = cv2.VideoCapture(0)
    cam.set(3, 640) # set video widht
    cam.set(4, 480) # set video height

    # Define min window size to be recognized as a face
    minW = 0.1*cam.get(3)
    minH = 0.1*cam.get(4)

    while True:

        ret, img =cam.read()
        # img = cv2.flip(img, -1) # Flip vertically

        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale( 
            gray,
            scaleFactor = 1.2,
            minNeighbors = 5,
            minSize = (int(minW), int(minH)),
        )

        for(x,y,w,h) in faces:

            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

            id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
            print(id)
        
            # Check if confidence is less them 100 ==> "0" is perfect match 
            if (confidence < 100):
                name_id = names[id]
                confidence = "  {0}%".format(round(100 - confidence))
                
            else:
                name_id = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))
            
            update_database(name_id)
            
            cv2.putText(img, str(name_id), (x+5,y-5), font, 1, (255,255,255), 2)
            cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  
        
        cv2.imshow('camera',img) 

        k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
        if k == 27:
            break

    # Do a bit of cleanup
    print("\n [INFO] Exiting Program and cleanup stuff")
    cam.release()
    cv2.destroyAllWindows()
