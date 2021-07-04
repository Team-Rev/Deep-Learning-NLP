from pptx import Presentation
import os

root_dir = os.listdir('CCNA')
ccna_version = []
for i in range(len(root_dir)):
    sub_dir = os.listdir('CCNA/'+f'{root_dir[i]}')
    for j in range(len(sub_dir)):
        prs = Presentation(f'CCNA/{root_dir[i]}/{sub_dir[j]}')
        for slide in prs.slides:
            for shape in slide.shapes:
                has_table = shape.has_table
                if shape.has_text_frame:
                    for paragraph in shape.text_frame.paragraphs:
                        print(paragraph.text)
                elif shape.has_table:
                    table = shape.table
                    print('테이블 시작')
                    for row_idx, row in enumerate(table.rows):
                        for col_idx, cell in enumerate(row.cells):
                            print(cell.text + ' | ', end='')
                        print()
                    print('테이블 종료')








# for slide in prs.slides:
#     for shape in slide.shapes:
#         if not shape.has_text_frame:
#             continue
#         for paragraph in shape.text_frame.paragraphs:
#             result.append(paragraph.text)
