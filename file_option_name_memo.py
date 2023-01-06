file_name_option_dict = {
    "full" : [[], [], [], [], [], []],
    "color=half_or=left" : [[3,7,9], [3,7,9], [3,7,9], [], [], [8,9,10,11,12,13,14]],
    "one_view_point_3" : [[], [], [], [], [], [0,1,2,4,5,6,7,8,9,10,11,12,13,14]],
    "one_view_point_11": [[], [], [], [], [], [0,1,2,3,4,5,6,7,8,9,10,12,13,14]],
    "one_view_half_scale": [[3,7,9], [3,7,9], [3,7,9], [0,2,4,6], [], [0,1,2,4,5,6,7,8,9,10,11,12,13,14]],
    "two_view_point_34" : [[], [], [], [], [], [0,1,2,5,6,7,8,9,10,11,12,13,14]],
    "half_view_point" : [[], [], [], [], [], [0,1,2,3,4,5,6]],
    "all4_factors_veiw3":[[1,2,4,5,7,8], [1,2,4,5,7,8], [1,2,4,5,7,8], [1,3,5,7], [], [0,1,2,4,5,6,7,8,9,10,11,12,13,14]],
    "one_view_point_88884" : [[2,7], [2,7], [2,7], [], [], [0,1,2,4,5,6,7,8,9,10,11,12,13,14]],
    "one_view_point_55544": [[0,2,4,6,8], [0,2,4,6,8], [0,2,4,6,8], [1,3,4,6], [], [0,1,2,4,5,6,7,8,9,10,11,12,13,14]],
    "five_view_point_55544": [[0,2,4,6,8], [0,2,4,6,8], [0,2,4,6,8], [1,3,4,6], [], [5,6,7,8,9,10,11,12,13,14]],
    "three_view_point_88844" : [[2,7], [2,7], [2,7], [1,3,4,6], [], [3,4,5,6,7,8,9,10,11,12,13,14]],
}
color_list = ["red","orange","yellow","light green","emerald green","cyan","symphony blue","blue","purple","pink"]
size_list = [f"size={s}" for s in range(1,9)]
shape_list = ["cube","cylinder","ball","pole"]
orientation_list = [o for o in range(15)]
label_word_correspondance = [[color+" floor" for color in color_list],[color+" wall" for color in color_list],[color+" object" for color in color_list],size_list,shape_list,orientation_list]