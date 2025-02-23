import numpy as np
from math import atan2
import numpy as np
import torch 

# Tìm hình chiếu của hộp
def projection(bin):

    height, length, width= bin.shape
    pj = np.zeros((length, width), dtype=int)
    
    for i in range(length):
        for j in range(width):
            max_height = -1
            for k in range(height):
                if bin[k, i, j] == 1:
                    max_height = k
            pj[i, j] = max_height + 1
    
    return pj

# Tạo bao lồi
def graham_scan(points):
    # Tìm điểm thấp nhất và trái nhất
    if points == []:
        return [(0,0)]
    start = min(points, key=lambda p: (p[1], p[0]))
    points.pop(points.index(start))
    
    # Sắp xếp các điểm theo góc so với điểm khởi đầu
    points.sort(key=lambda p: (atan2(p[1] - start[1], p[0] - start[0]), (p[0] - start[0]) ** 2 + (p[1] - start[1]) ** 2))
    
    # Duyệt qua các điểm để xây dựng bao lồi
    hull = [start]
    for point in points:
        while len(hull) > 1 and ccw(hull[-2], hull[-1], point) <= 0:
            hull.pop()
        hull.append(point)
    return hull

def ccw(p1, p2, p3):
    # Tính diện tích tứ giác, nếu > 0 thì ngược chiều kim đồng hồ, = 0 thì thẳng hàng, < 0 thì theo chiều kim đồng hồ
    return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])

def is_point_in_polygon(polygon, point):
    x, y = point
    n = len(polygon)
    inside = False
    
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        
        # Kiểm tra điểm có nằm trên cạnh không
        if (x1 <= x <= x2 or x2 <= x <= x1) and (y1 <= y <= y2 or y2 <= y <= y1):
            if (x - x1) * (y2 - y1) == (y - y1) * (x2 - x1):
                return False  # Điểm nằm trên cạnh
        
        # Thuật toán Ray Casting
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1):
            inside = not inside
    
    return inside
    # Kiểm tra trọng tâm nằm trong bao lồi không

def is_point_in_convex_hull(points, point):
    hull = graham_scan(points)
    return is_point_in_polygon(hull, point)

def Stable(bin: torch.tensor,box: list,i: int ,j : int ,z : int):
    mat=projection(bin)

    print(mat)

    length, width = mat.shape 
    l, w = int(box[0]), int(box[1])

    max_value=np.amax(mat[i:i+l,j:j+w])
    if max_value!=z: 
        return False 

    indices = np.where(mat[i:i+l,j:j+w] == max_value)
    points = list(zip(indices[0]+i, indices[1]+j))
    list_points = []
    for p in points:
        list_points.append(p)
        if p[0] < length: 
            list_points.append((p[0] + 1, p[1]))
        if p[1] < width: 
            list_points.append((p[0], p[1] + 1))
        if (p[0] < length) and (p[1] < width): 
            list_points.append((p[0] + 1, p[1] + 1)) 
    list_points=list(set(list_points)) 
    # Kiểm tra thỏa mãn điều kiện vật lý không 
    return is_point_in_convex_hull(list_points,(i+l/2,j+w/2)) 

