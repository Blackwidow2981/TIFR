from flask import Flask,render_template,request
import cv2
import numpy as np
import math
import torch
import os
import time
from PIL import Image, ImageDraw
from sympy import symbols, Eq, solve
from PIL import Image
import PIL
from ultralytics import YOLO

# file saved in: C:\Users\AS\Desktop\TIFR\Integration_BE\TIFR

app=Flask(__name__)
@app.route("/")
@app.route("/home")
def home():
    return render_template('index.html')
@app.route('/contact',methods=['POST','GET'])
def contact():
    # C:/Users/AS/Desktop/TIFR/813 pictures/Image2.jpg
    output= request.form.to_dict()
    print(output)
    name=output["flag"]
    path =name
    # path = "/content/Image10.jpg"
    print(path)
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    x_cord = 0
    y_cord = 0
    radius = 0
    xaxis = 0
    yaxis = 0
    radq = 0
    # Convert to grayscale.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Blur using 3 * 3 kernel.
    gray_blurred = cv2.blur(gray, (3, 3))

    # Apply Hough transform on the blurred image.
    detected_circles = cv2.HoughCircles(gray_blurred,
                                        cv2.HOUGH_GRADIENT, 1, 20, param1=50,
                                        param2=30, minRadius=100, maxRadius=110)

    # Draw circles that are detected.
    if detected_circles is not None:

        # Convert the circle parameters a, b and r to integers.
        detected_circles = np.uint16(np.around(detected_circles))

        for pt in detected_circles[0, :]:
            a, b, r = pt[0], pt[1], pt[2]
            # print(a)
            x_cord = a
            xaxis = a
            # print(b)
            y_cord = b
            yaxis = b
            # print(r)
            radius = r
            radq = r
            # Draw the circumference of the circle.
            cv2.circle(img, (a, b), r, (0, 255, 0), 2)

            # Draw a small circle (of radius 1) to show the center.
            cv2.circle(img, (a, b), 1, (0, 0, 255), 3)
    # print(f"{x_cord} {y_cord} {radius}")
    print(f"The center coordinates are: {x_cord},{y_cord}")
    print(f"The radius of the circle is: {radius}")
    if ( x_cord == 0 and y_cord ) or radius == 0:
        print("The first circle is not detected###########################################################")
        return ""                         #----------------------------------------------have to add return statements
    img = Image.open(path)
    box = (x_cord - radius - 10, y_cord - radius - 10, x_cord + radius + 15, y_cord + radius + 10)
    img2 = img.crop(box)
    img2.save(f'C:/Users/sneha/Downloads/TIFR/TIFR/Images/myimage_cropped.jpg')


    image = cv2.imread(f"C:/Users/sneha/Downloads/TIFR/TIFR/Images/myimage_cropped.jpg")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Use canny edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Apply HoughLinesP method to
    # to directly obtain line end points
    lines_list = []
    lines = cv2.HoughLinesP(
        edges,  # Input edge image
        1,  # Distance resolution in pixels
        np.pi / 180,  # Angle resolution in radians
        threshold=50,  # Min number of votes for valid line
        minLineLength=50,  # Min allowed length of line
        maxLineGap=10  # Max allowed gap between line for joining them
    )
    list_slope = []
    print(lines)
    # Iterate over points
    for points in lines:
        x1, y1, x2, y2 = points[0]
        slope = (y2 - y1) / (x2 - x1)
        slope = "{:.1f}".format(slope)
        list_slope.append(slope)
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Maintain a simples lookup list for points
        lines_list.append([(x1, y1), (x2, y2)])
        # cv2.imshow(image)
        break

    for points in lines:
        # Extracted points nested in the list
        x1, y1, x2, y2 = points[0]
        slope = (y2 - y1) / (x2 - x1)
        slope = "{:.1f}".format(slope)
        if len(list_slope) == 3:
            break
        if slope not in list_slope:
            # Draw the lines joing the points
            # On the original image
            x = 0
            for s in list_slope:
                Float = float(s)
                if s == 'inf' or s == '-inf':
                    continue
                start = Float - 0.5
                end = Float + 0.5
                print(list_slope)
                range_1 = np.arange(start, end, 0.1)
                if slope in range_1:
                    x = 1
                    break
                else:
                    x = 0
            if x == 0:
                list_slope.append(slope)
                cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # cv2.imshow(image)
                # Maintain a simples lookup list for points
                lines_list.append([(x1, y1), (x2, y2)])

    print("hello")
    # cv2.imwrite("/content/h.jpg", image)
    # print(list_slope)
    # print(lines_list)
    if len(lines_list) < 3 or len(list_slope) < 3:
        print("Only two lines detected. Please check the image quality##########################################")
        return ""                         #----------------------------------------------have to add return statements
    deg_slope=[]
    for slp in list_slope:
        # print(type(slp))
        deg = math.degrees(math.atan(float(slp)))
        if deg == -90.0 or deg == 90.0:
            deg_slope.append(90.0)
            continue
        if deg == 0.0:
            deg_slope.append(0.0)
            continue
        if deg > 0.0:
            deg_slope.append(deg+180)
        else:
            deg_slope.append(deg+360)
    deg_slope.sort()
    print(deg_slope)
    dec = deg_slope[0]
    inc = deg_slope[0]
    flag = True
    for slp in deg_slope:
        # print(dec,"sadsda",inc)
        if slp < dec or slp > inc:
            flag = False
            break
        dec = slp + 115
        inc = slp + 125
    if flag == False:
        print("Slopes not aligning well. Check the positioning")
        return ""                  #----------------------------------------------have to add return statements
    o = 1
    lengthDict={}
    # Length of the lines
    for points in lines_list:
        x1, y1 = points[0]
        x2, y2 = points[1]
        length = math.sqrt(math.pow((x2 - x1), 2) + math.pow((y2 - y1), 2))
        length = "{:.2f}".format(length)
        lengthDict[f"line{o}"]=length
        print(f'Length of line{o} {length}')
        o = o + 1
    print()

    # Intersection point
    x, y = symbols('x y')
    list_inter = []
    i = -1
    for point in list_slope:
        i = i + 1
        if point == '-inf' or point == 'inf':
            continue
        else:
            l = [lines_list[i][0][0], lines_list[i][0][1], lines_list[i][1][0], lines_list[i][1][1]]
            list_inter.append(l)
    print(list_inter)

    eq1 = ()
    eq2 = ()
    for p in list_inter:
        slope = (p[3] - p[1]) / (p[2] - p[0])
        slope = "{:.2f}".format(slope)
        # print(float(slope))
        slope = float(slope)
        sub = (float(slope) * float(p[0])) - float(p[1])
        # print(f'Equation of line is: y-({slope}x)+{sub}=0')
        eq1 = Eq(y - (slope * x) + sub)
        break
    j = 0
    for p in list_inter:
        if j == 0:
            j = j + 1
            continue
        slope = (p[3] - p[1]) / (p[2] - p[0])
        slope = "{:.2f}".format(slope)
        # print(float(slope))
        slope = float(slope)
        sub = (float(slope) * float(p[0])) - float(p[1])
        # print(f'Equation of line is: y-({slope}x)+{sub}=0')
        eq2 = Eq(y - (slope * x) + sub)

    print(eq1)
    print(eq2)
    print()
    a = solve((eq1, eq2), (x, y))[x]
    p = "{:.0f}".format(a)
    b = solve((eq1, eq2), (x, y))[y]
    q = "{:.0f}".format(b)
    print(f'Co-ordinates of intersection of the arms is: (x: {p}, y: {q})')
    print(f'The center of the circle is: (x: {x_cord}, y: {y_cord})')
    # print(a)
    # print(b)
    # 308 256
    intersectCoord={"X":p,"Y":q}
    circleCoord={"X":x_cord,"Y":y_cord}
    innerCord = [int(p), int(q)]
   

    x_cord = 0
    y_cord = 0
    radius = 0
    # Convert to grayscale.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Blur using 3 * 3 kernel.
    gray_blurred = cv2.blur(gray, (3, 3))

    # Apply Hough transform on the blurred image.
    detected_circles = cv2.HoughCircles(gray_blurred,
                                        cv2.HOUGH_GRADIENT, 1, 20, param1=50,
                                        param2=30, minRadius=100, maxRadius=110)

    # Draw circles that are detected.
    if detected_circles is not None:

        # Convert the circle parameters a, b and r to integers.
        detected_circles = np.uint16(np.around(detected_circles))

        for pt in detected_circles[0, :]:
            a, b, r = pt[0], pt[1], pt[2]
            # print(a)
            x_cord = a
            # print(b)
            y_cord = b
            # print(r)
            radius = r
            # Draw the circumference of the circle.
            cv2.circle(image, (a, b), r, (0, 255, 0), 2)

            # Draw a small circle (of radius 1) to show the center.
            cv2.circle(image, (a, b), 1, (0, 0, 255), 3)
            # cv2_imshow(img)
            # cv2.waitKey(0)

    cv2.circle(image, (int(p), int(q)), 1, (0, 0, 255), 6)
    cv2.circle(image, (int(x_cord), int(y_cord)), 1, (0, 0, 255), 5)
    cv2.circle(image, (int(x_cord), int(y_cord)), radius, (0, 0, 255), 2)
    
    outerCord = [int(x_cord), int(y_cord)]
    cv2.line(image,innerCord,outerCord,(255,0,0),2)
    dist = math.dist(innerCord, outerCord)
    print(f"The offset of the center of circles and intersection point is: {dist}")
    # cv2.imshow(image)
    # cv2.imshow('hello', img)
    # print(x)
    pq='C:/Users/sneha/Downloads/TIFR/TIFR/Images/myimage.jpg'
    cv2.imwrite(pq, image)
    im1 = Image.open(path)
    im2 = Image.open(pq)
    back_im = im1.copy()
    back_im.paste(im2, (xaxis - radq - 10, yaxis - radq - 10))
    back_im.save('C:/Users/sneha/Downloads/TIFR/TIFR/Images/Image2.jpg', quality=95)
    # cv2.waitKey(0)
    thdict={"lineLength":lengthDict,"Intersection":intersectCoord,"Circle":circleCoord, "offset":dist}
    return render_template("results2.html" , name=thdict)

# path= C:/Users/AS/Desktop/TIFR/HexaboardImages/Hexaboard_1/Hole1.jpg

@app.route('/concentric',methods=['POST','GET'])
def concentric():
    if(os.path.exists("C:/Users/sneha/Downloads/TIFR/TIFR/static/Hole.jpg")):
        os.remove("C:/Users/sneha/Downloads/TIFR/TIFR/static/Hole.jpg/image0.jpg")
        os.rmdir("C:/Users/sneha/Downloads/TIFR/TIFR/static/Hole.jpg")


    if (os.path.exists("C:/Users/sneha/Downloads/TIFR/TIFR/runs/segment/predict")):
        os.remove("C:/Users/sneha/Downloads/TIFR/TIFR/runs/segment/predict/image0.jpg")
        os.rmdir("C:/Users/sneha/Downloads/TIFR/TIFR/runs/segment/predict")
    output= request.form.to_dict()
    print(output)
    name=output["flag"]
    path = name
    # path = f"C:/Users/AS/Desktop/TIFR/HexaboardImages/Hexaboard_1/Hole11.jpg"
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    # cv2.imshow(img)
    print(path)
    innerXCord = 0
    innerYCord = 0
    innerRadius = 0
    # Convert to grayscale.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Blur using 3 * 3 kernel.
    gray_blurred = cv2.blur(gray, (3, 3))

    # Apply Hough transform on the blurred image.
    # detected_circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, 1, 20, param1 = 50, param2 = 25, minRadius = 170,maxRadius=180)
    detected_circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, 1, 21, param1=50, param2=20, minRadius=168,maxRadius=180)
    # 170 to 180 standard (inner circle)
    # 320 to 350 standard (outer circle (only the first circle is the required))
    # 210 to 220 magnified (inner circle)

    # Draw circles that are detected.
    if detected_circles is not None:

        detected_circles = np.uint16(np.around(detected_circles))

        for pt in detected_circles[0, :]:
            a, b, r = pt[0], pt[1], pt[2]
            innerXCord=a
            innerYCord=b
            innerRadius=r
            # Draw the circumference of the circle.
            cv2.circle(img, (a, b), r, (0, 255, 0), 2)

          # Draw a small circle (of radius 1) to show the center.
            cv2.circle(img, (a, b), 1, (0, 0, 255), 3)
            # cv2.imshow(img)
            # cv2.waitKey(0)
            break


    img = cv2.imread(path, cv2.IMREAD_COLOR)

    outerXCord = 0
    outerYCord = 0
    outerRadius = 0
    # Convert to grayscale.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Blur using 3 * 3 kernel.
    gray_blurred = cv2.blur(gray, (3, 3))

    # Apply Hough transform on the blurred image.
    # detected_circles = cv2.HoughCircles(gray_blurred,cv2.HOUGH_GRADIENT, 1, 20, param1=50,param2=30, minRadius=320, maxRadius=350)
    detected_circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=320, maxRadius=350)
    # 170 to 180 standard (inner circle)
    # 320 to 350 standard (outer circle (only the first circle is the required))
    # 210 to 220 magnified (inner circle)
    # print(detected_circles)
    # Draw circles that are detected.
    innerCord = [innerXCord, innerYCord]
    # Draw circles that are detected.
    dist = math.inf
    outerXCord = math.inf
    outerYCord = math.inf
    outerRadius = math.inf
    if detected_circles is not None:

        # Convert the circle parameters a, b and r to integers.
        detected_circles = np.uint16(np.around(detected_circles))
        # x = 0

        for pt in detected_circles[0, :]:
            # if x==1:
            a, b, r = pt[0], pt[1], pt[2]
            outerCord = [a, b]
            d = math.dist(innerCord, outerCord)
            # print(d,dist)
            if d < dist:
                dist = d
                outerXCord = a
                outerYCord = b
                outerRadius = r + 10
            # Draw the circumference of the circle.
            cv2.circle(img, (a, b), r + 10, (0, 255, 0), 2)

            # Draw a small circle (of radius 1) to show the center.
            cv2.circle(img, (a, b), 1, (0, 0, 255), 3)
            # cv2.imshow("hello",img)
            # break
            # cv2.waitKey(0)
        # x=x+1

    print(f"{innerRadius}: inner radius and {outerRadius}: outer radius")
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    cv2.circle(img, (outerXCord, outerYCord), outerRadius, (0, 255, 0), 2)
    cv2.circle(img, (outerXCord, outerYCord), 1, (0, 0, 255), 3)
    cv2.circle(img, (innerXCord, innerYCord), innerRadius, (0, 255, 0), 2)
    cv2.circle(img, (innerXCord, innerYCord), 1, (0, 0, 255), 3)
    # cv2.imshow(img)
    print(f"The X and Y coordinates outer circle are: {outerXCord, outerYCord}")
    print(f"The X and Y coordinates inner circle are: {innerXCord, innerYCord}")
    innerCord = [innerXCord, innerYCord]
    outerCord = [outerXCord, outerYCord]
    dist = math.dist(innerCord, outerCord)
    print(f"The offset of the concentric circles is: {dist}")
    p = "C:/Users/sneha/Downloads/TIFR/TIFR/Images/Hole.jpg"
    # C:\Users\AS\Desktop\TIFR\Final Outputs\Concentric/Board1
    cv2.imwrite(p, img)

    path = 'C:/Users/sneha/Downloads/TIFR/forLoop/best.pt'
    path2 = 'C:/Users/sneha/Downloads/TIFR/localYOLO/best.pt'
    print(path)
    model2 = torch.hub.load('ultralytics/yolov5', 'custom', path2, force_reload=False)
    model = YOLO(path)
    img = cv2.imread(p)
    result2 = model2(img)
    u = result2.pandas().xyxy[0].name
    temp={}
    x=0
    for a in u:
        temp[f"Pad{x}"] = a
        # print(a)
        x = x + 1
    res = model.predict(img, save=True, imgsz=640, conf=0.128, verbose=False)
    print(temp)
    print("-----")
    areaContour=[]
    cn=0
    for q in range(len(res[0].masks.data)):
        if cn==3:
            break
        m = (res[0].masks.data[q].cpu().numpy() * 255).astype("uint8")
        ret, thresh = cv2.threshold(m, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        edged = cv2.Canny(m, 30, 200)
        contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # print(len(contours))
        area = cv2.contourArea(contours[0])
        areaContour.append(area)
        cn=cn+1
        # print(area)
    print(areaContour)
    picture = Image.open(r'C:/Users/sneha/Downloads/TIFR/TIFR/runs/segment/predict/image0.jpg')
    picture = picture.save(f"C:/Users/sneha/Downloads/TIFR/TIFR/static/Image1.jpg")

    thdict={"OrientationSignalPad":temp,"AreaOfPad":areaContour,"Offset":dist}

    return render_template('results1.html',hi=thdict)

@app.route("/indexstage1",methods=['POST','GET'])
def indextstage1():
    return render_template('indexstage1.html')

@app.route("/indexstage2",methods=['POST','GET'])
def indextstage2():
    return render_template('indexstage2.html')

@app.route("/exit",methods=['POST','GET'])
def exit():
    return render_template('index.html')    

@app.route("/survey1",methods=['POST','GET'])
def survey1():
    return render_template('survey1.html')  

@app.route("/survey2",methods=['POST','GET'])
def survey2():
    return render_template('survey2.html')    



if __name__=='__main__':
    app.run(debug=True,port=5555)
