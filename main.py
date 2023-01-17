# importing necessary libraries
import cv2 as opencv
import numpy as np



def displayOutputs():
    opencv.imshow("Detected characters", src_img)
    opencv.imshow("Preprocessed image", bin_img)
    opencv.imshow("Segmenting characters", final_thr)
    opencv.waitKey(0)
    opencv.destroyAllWindows()

def array_for_lines(array):
    list_up = []
    list_low = []
    for y in range(5, len(array) - 5):
        f_a, f_pr = foreline(y, array)
        l_a, l_pr = last_line(y, array)
        print(str(f_a) + ',' + str(f_pr) + ',' + str(l_a) + ',' + str(l_pr) + ',' + str(y))
        if f_a >= 7 and f_pr >= 5:
            list_up.append(y)
        if f_a >= 5 and f_pr >= 7:
            list_low.append(y)
    return list_up, list_low


def foreline(y, array):
    count_front = 0
    count_back = 0
    for i in array[y:y + 10]:
        if i > 3:
            count_front += 1
    for i in array[y - 10:y]:
        if i == 0:
            count_back += 1
    return count_front, count_back


def last_line(y, array):
    count_front = 0
    count_back = 0
    for i in array[y:y + 10]:
        if i == 0:
            count_front += 1
    for i in array[y - 10:y]:
        if i > 3:
            count_back += 1
    return count_front, count_back


def last_line_word(y, array, a):
    count_front = 0
    count_back = 0
    for i in array[y:y + 2 * a]:
        if i < 2:
            count_front += 1
    for i in array[y - a:y]:
        if i > 2:
            count_back += 1
    return count_back, count_front


def last_line_array(array, a):
    list_last_lines = []
    for y in range(len(array)):
        l_pr, l_a = last_line_word(y, array, a)
        if l_a >= int(0.8 * a) and l_pr >= int(0.7 * a):
            list_last_lines.append(y)
    return list_last_lines


def refine_last_word(array):
    ref_list = []
    for y in range(len(array) - 1):
        if array[y] + 1 < array[y + 1]:
            ref_list.append(array[y])
    ref_list.append(array[-1])
    return ref_list


def refine_entire_array(array_upper, array_lower):
    upp_lines = []
    lowr_lines = []
    for y in range(len(array_upper) - 1):
        if array_upper[y] + 5 < array_upper[y + 1]:
            upp_lines.append(array_upper[y] - 10)
    for y in range(len(array_lower) - 1):
        if array_lower[y] + 5 < array_lower[y + 1]:
            lowr_lines.append(array_lower[y] + 10)

    upp_lines.append(array_upper[-1] - 10)
    lowr_lines.append(array_lower[-1] + 10)

    return upp_lines, lowr_lines


def letter_dimension(contours):
    letter_breadth_sum = 0
    count = 0
    for cnt in contours:
        if opencv.contourArea(cnt) > 20:
            x, y, w, h = opencv.boundingRect(cnt)
            letter_breadth_sum += w
            count += 1
    return letter_breadth_sum / count


def detect_last_word(endmost_local, i, binary_output):
    cnt_y = np.zeros(shape=width)
    for x in range(width):
        for y in range(endmost_local[i], endmost_local[i + 1]):
            if binary_output[y][x] == 255:
                cnt_y[x] += 1

    contours, hierarchy = opencv.findContours(lines_img[i], opencv.RETR_EXTERNAL, opencv.CHAIN_APPROX_SIMPLE)
    letter_breadth_sum = 0
    count = 0
    # getting the mean width
    for cnt in contours:
        if opencv.contourArea(cnt) > 20:
            x, y, w, h = opencv.boundingRect(cnt)
            letter_breadth_sum += w
            count += 1
    if count != 0:
        mean_breadth = letter_breadth_sum / count
    else:
        mean_breadth = 0
    spaces = []
    line_tail = []
    for x in range(len(cnt_y)):
        number = int(0.5 * int(mean_breadth)) - np.count_nonzero(
            cnt_y[x - int(0.25 * int(mean_breadth)):x + int(0.25 * int(mean_breadth))])
        if max(cnt_y[0:x + 1]) >= 3 and number >= 0.4 * int(mean_breadth):
            spaces.append(x)
        if max(cnt_y[x:]) <= 2:
            line_tail.append(x)
    tline_end = min(line_tail) + 10

    ret = []
    fin_spaces = []
    for j in range(len(spaces)):
        if spaces[j] < tline_end:
            if spaces[j] == spaces[j - 1] + 1:
                ret.append(spaces[j - 1])
            elif spaces[j] != spaces[j - 1] + 1 and spaces[j - 1] == spaces[j - 2] + 1:
                ret.append(spaces[j - 1])
                retavg = int(sum(ret) / len(ret))
                fin_spaces.append(retavg)
                ret = []
            elif spaces[j] != spaces[j - 1] + 1 and spaces[j - 1] != spaces[j - 2] + 1 and spaces[j] != spaces[
                j + 1] - 1:
                fin_spaces.append(spaces[j])
        elif spaces[j] == tline_end:
            fin_spaces.append(tline_end)
    for x in fin_spaces:
        final_thr[fin_loc[i]:fin_loc[i + 1], x] = 255
    return fin_spaces


def character_segmentation(lines_img, x_lines, i):
    copy_img = lines_img[i].copy()
    x_linescopy = x_lines[i].copy()

    letter_k = []

    contours, hierarchy = opencv.findContours(copy_img, opencv.RETR_EXTERNAL, opencv.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if opencv.contourArea(cnt) > 5:
            x, y, w, h = opencv.boundingRect(cnt)
            letter_k.append((x, y, w, h))

    letter_breadth_sum = 0
    count = 0
    for cnt in contours:
        if opencv.contourArea(cnt) > 20:
            x, y, w, h = opencv.boundingRect(cnt)
            letter_breadth_sum += h
            count += 1

    letter = sorted(letter_k, key=lambda student: student[0])

    for e in range(len(letter)):
        if e < len(letter) - 1:
            if abs(letter[e][0] - letter[e + 1][0]) <= 2:
                x, y, w, h = letter[e]
                x2, y2, w2, h2 = letter[e + 1]
                if h >= h2:
                    letter[e] = (x, y2, w, h + h2)
                    letter.pop(e + 1)
                elif h < h2:
                    letter[e + 1] = (x2, y, w2, h + h2)
                    letter.pop(e)

    for e in range(len(letter)):
        letter_img_tmp = lines_img[i][letter[e][1] - 0:letter[e][1] + letter[e][3] + 0,
                         letter[e][0] - 0:letter[e][0] + letter[e][2] + 0]
        letter_img_tmp = opencv.resize(letter_img_tmp, dsize=(28, 28), interpolation=opencv.INTER_AREA)
        width = letter_img_tmp.shape[1]
        height = letter_img_tmp.shape[0]
        count_y = np.zeros(shape=(width))
        for x in range(width):
            for y in range(height):
                if letter_img_tmp[y][x] == 255:
                    count_y[x] = count_y[x] + 1
                    # check the length of the x_linescopy list before attempting to access an element of it.
                    if len(x_linescopy) > 0:
                        if letter[e][0] < x_linescopy[0]:
                            x_linescopy.insert(0, letter[e][0])

    x_linescopy.pop(0)
    word = 1
    letter_index = 0
    for e in range(len(letter)):
        if letter[e][0] < x_linescopy[0]:
            letter_index += 1
            letter_img_tmp = lines_img[i][letter[e][1] - 0:letter[e][1] + letter[e][3] + 5,
                             letter[e][0] - 2:letter[e][0] + letter[e][2] + 2]
            char_img = opencv.resize(letter_img_tmp, dsize=(28, 28), interpolation=opencv.INTER_AREA)
            opencv.imwrite('./segmented/photo1/' + str(i + 1) + '' + str(word) + '' + str(letter_index) + '.jpg',
                           255 - char_img)
        else:
            x_linescopy.pop(0)
            word += 1
            char_index = 1
            letter_img_tmp = lines_img[i][letter[e][1] - 0:letter[e][1] + letter[e][3] + 5,
                             letter[e][0] - 2:letter[e][0] + letter[e][2] + 2]
            char_img = opencv.resize(letter_img_tmp, dsize=(28, 28), interpolation=opencv.INTER_AREA)
            opencv.imwrite('./segmented/photo1/' + str(i + 1) + '' + str(word) + '' + str(char_index) + '.jpg',
                           255 - char_img)


print("\n---Program launched successfully---\n")
src_img = opencv.imread('./handwritten.jpg', 1)
copy = src_img.copy()
height = src_img.shape[0]
width = src_img.shape[1]

# Resizing the image
src_img = opencv.resize(copy, dsize=(500, int(500 * height / width)), interpolation=opencv.INTER_AREA)
height = src_img.shape[0]
width = src_img.shape[1]
# gray scaling the image
grey_img = opencv.cvtColor(src_img, opencv.COLOR_BGR2GRAY)

# Applying adaptive threshold with the kernel
bin_img = opencv.adaptiveThreshold(grey_img, 255, opencv.ADAPTIVE_THRESH_MEAN_C, opencv.THRESH_BINARY_INV, 21, 20)
coords = np.column_stack(np.where(bin_img > 0))
angle = opencv.minAreaRect(coords)[-1]
if angle < -45:
    angle = -(90 + angle)
else:
    angle = -angle
h = bin_img.shape[0]
w = bin_img.shape[1]
center = (w // 2, h // 2)
angle = 0
M = opencv.getRotationMatrix2D(center, angle, 1.0)
bin_img = opencv.warpAffine(bin_img, M, (w, h),
                            flags=opencv.INTER_CUBIC, borderMode=opencv.BORDER_REPLICATE)

bin_img1 = bin_img.copy()
bin_img2 = bin_img.copy()
# structuring the image contents
kernel = opencv.getStructuringElement(opencv.MORPH_ELLIPSE, (3, 3))
kernel1 = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=np.uint8)

print("Noise elimination successful!")
final_thr = opencv.morphologyEx(bin_img, opencv.MORPH_CLOSE, kernel)
contr_retrival = final_thr.copy()

print("Character Segmentation Started...")
count_x = np.zeros(shape=height)
for y in range(height):
    for x in range(width):
        if bin_img[y][x] == 255:
            count_x[y] = count_x[y] + 1

loc_min = []
for y in range(len(count_x)):
    if 10 <= y <= len(count_x) - 11:
        arr1 = count_x[y - 10:y + 10]
    elif y < 10:
        arr1 = count_x[0:y + 10]
    else:
        arr1 = count_x[y - 10:len(count_x) - 1]
    if min(arr1) == count_x[y]:
        loc_min.append(y)

fin_loc = []
init = []
end = []
for z in range(len(loc_min)):
    if z != 0 and z != len(loc_min) - 1:
        if loc_min[z] != (loc_min[z - 1] + 1) and loc_min[z] != (loc_min[z + 1] - 1):
            fin_loc.append(loc_min[z])
        elif loc_min[z] != (loc_min[z - 1] + 1) and loc_min[z] == (loc_min[z + 1] - 1):
            init.append(loc_min[z])
        elif loc_min[z] == (loc_min[z - 1] + 1) and loc_min[z] != (loc_min[z + 1] - 1):
            end.append(loc_min[z])
    elif z == 0:
        if loc_min[z] != (loc_min[z + 1] - 1):
            fin_loc.append(loc_min[z])
        elif loc_min[z] == (loc_min[z + 1] - 1):
            init.append(loc_min[z])
    elif z == len(loc_min) - 1:
        if loc_min[z] != (loc_min[z - 1] + 1):
            fin_loc.append(loc_min[z])
        elif loc_min[z] == (loc_min[z - 1] + 1):
            end.append(loc_min[z])
for j in range(len(init)):
    mid = (init[j] + end[j]) / 2
    if (mid % 1) != 0:
        mid = mid + 0.5
    fin_loc.append(int(mid))
fin_loc = sorted(fin_loc)
no_of_lines = len(fin_loc) - 1

# checking whether the image contains multiple lines
lines_img = []
# checking average width of each letter to draw contours
for i in range(no_of_lines):
    lines_img.append(bin_img2[fin_loc[i]:fin_loc[i + 1], :])
contours, hierarchy = opencv.findContours(contr_retrival, opencv.RETR_EXTERNAL, opencv.CHAIN_APPROX_SIMPLE)
final_contr = np.zeros((final_thr.shape[0], final_thr.shape[1], 3), dtype=np.uint8)
opencv.drawContours(src_img, contours, -1, (0, 255, 0), 1)
mean_lttr_width = letter_dimension(contours)

x_lines = []
# detect lines in the handwritten image
for i in range(len(lines_img)):
    x_lines.append(detect_last_word(fin_loc, i, bin_img))

for i in range(len(x_lines)):
    x_lines[i].append(width)

# segmenting letters in the word
opencv.waitKey(0)
for i in range(no_of_lines):
    character_segmentation(lines_img, x_lines, i)

chr_img = bin_img1.copy()

contours, hierarchy = opencv.findContours(chr_img, opencv.RETR_EXTERNAL, opencv.CHAIN_APPROX_SIMPLE)

for cont in contours:
    if opencv.contourArea(cont) > 20:
        x, y, w, h = opencv.boundingRect(cont)
        opencv.rectangle(src_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
# showing the outputs
displayOutputs()