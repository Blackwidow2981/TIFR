import cv2
import numpy as np
import math
import time
from PIL import Image, ImageDraw
from sympy import symbols, Eq, solve

# Read image.
m = 1
for m in range(10):
    m = m+1
    # print(m,"fe")
    path = 'C:/Users/AS/Desktop/TIFR/813 pictures/Image{}.jpg'.format(m)
    # path = "/content/Image10.jpg"
    print(path)
    if m==1 or m==6:
        continue
    if path == 'C:/Users/AS/Desktop/TIFR/813 pictures/Image9.jpg' or path == 'C:/Users/AS/Desktop/TIFR/813 pictures/Image42.jpg' or path == 'C:/Users/AS/Desktop/TIFR/813 pictures/Image66.jpg':
        continue
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    x_cord = 0
    y_cord = 0
    radius = 0
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
            # print(b)
            y_cord = b
            # print(r)
            radius = r
            # Draw the circumference of the circle.
            cv2.circle(img, (a, b), r, (0, 255, 0), 2)

            # Draw a small circle (of radius 1) to show the center.
            cv2.circle(img, (a, b), 1, (0, 0, 255), 3)
    # print(f"{x_cord} {y_cord} {radius}")
    print(f"The center coordinates are: {x_cord},{y_cord}")
    print(f"The radius of the circle is: {radius}")
    img = Image.open(path)
    box = (x_cord - radius - 10, y_cord - radius - 10, x_cord + radius + 15, y_cord + radius + 10)
    img2 = img.crop(box)
    img2.save(f'D:/Python/images/myimage_cropped{m}.jpg')


    time.sleep(5)
    image = cv2.imread(f"D:/Python/images/myimage_cropped{m}.jpg")
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

    o = 1
    # Length of the lines
    for points in lines_list:
        x1, y1 = points[0]
        x2, y2 = points[1]
        length = math.sqrt(math.pow((x2 - x1), 2) + math.pow((y2 - y1), 2))
        length = "{:.2f}".format(length)
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

    # time.sleep(10)

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
    # cv2.imshow(image)
    # cv2.imshow('hello', img)
    # print(x)
    p = f"D:/Python/finalMerc/Image{m}.jpg"
    cv2.imwrite(p, image)
    # cv2.waitKey(0)
