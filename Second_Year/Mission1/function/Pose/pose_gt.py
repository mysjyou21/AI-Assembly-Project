import numpy as np
POSE_GT = {
'step1_a' : '000_000_000_left',
'step1_b' : '000_000_180_left',
'step2' : '090_000_180_left',
'step3' : '090_000_180_left',
'step4' : '090_000_180_left',
'step5' : '090_180_000_left',
'step6' : '090_180_270_left',
'step7' : '090_180_270_left',
'step8' : '090_180_270_left',
'part1' : '090_270_000_left', 
'part2' : '090_090_000_right',
'part3' : '090_090_000_right',
'part4' : '180_000_180_left',
'part5' : '090_000_180_left',
'part6' : '090_180_180_left',
}
POSE_TO_LABEL = {
'000_000_000_left' :  0,
'000_000_000_right' : 1,
'000_000_090_left' :  2,
'000_000_090_right' : 3,
'000_000_180_left' :  4,
'000_000_180_right' : 5,
'000_000_270_left' :  6,
'000_000_270_right' : 7,
'090_000_000_left' :  8,
'090_000_000_right' : 9,
'090_000_090_left' :  10,
'090_000_090_right' : 11,
'090_000_180_left' :  12,
'090_000_180_right' : 13,
'090_000_270_left' :  14,
'090_000_270_right' : 15,
'090_090_000_left' :  16,
'090_090_000_right' : 17,
'090_090_090_left' :  18,
'090_090_090_right' : 19,
'090_090_180_left' :  20,
'090_090_180_right' : 21,
'090_090_270_left' :  22,
'090_090_270_right' : 23,
'090_180_000_left' :  24,
'090_180_000_right' : 25,
'090_180_090_left' :  26,
'090_180_090_right' : 27,
'090_180_180_left' :  28,
'090_180_180_right' : 29,
'090_180_270_left' :  30,
'090_180_270_right' : 31,
'090_270_000_left' :  32,
'090_270_000_right' : 33,
'090_270_090_left' :  34,
'090_270_090_right' : 35,
'090_270_180_left' :  36,
'090_270_180_right' : 37,
'090_270_270_left' :  38,
'090_270_270_right' : 39,
'180_000_000_left' :  40,
'180_000_000_right' : 41,
'180_000_090_left' :  42,
'180_000_090_right' : 43,
'180_000_180_left' :  44,
'180_000_180_right' : 45,
'180_000_270_left' :  46,
'180_000_270_right' : 47,
}
# TODO
DUPLICATE_POSE_stefan_part1 = np.array([
[0, 40],
[1, 41],
[2, 42],
[3, 43],
[4, 44],
[5, 45],
[6, 46],
[7, 47],
[8, 28],
[9, 29],
[10, 30],
[11, 31],
[12, 24],
[13, 25],
[14, 26],
[15, 27],
[16, 20],
[17, 21],
[18, 22],
[19, 23],
[32, 36],
[33, 37],
[34, 38],
[35, 39],
	])
DUPLICATE_POSE_stefan_part2 = np.array([
[11, 27],
[12, 28],
[13, 29],
[14, 30],
[19, 35],
[20, 36],
[21, 37],
[22, 38],
[40, 44],
[41, 45],
[42, 46],
[43, 47],
	])
DUPLICATE_POSE_stefan_part3 = DUPLICATE_POSE_stefan_part2
