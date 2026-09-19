"""
MÃ NGUỒN PHÂN TÍCH KHẨU PHẦN ĂN & MỨC ĐÁP ỨNG RDA (TỐI ƯU CHO 21 BIẾN)
-------------------------------------------------------------------------
Đề tài: Đánh giá khẩu phần ăn và hành vi ăn uống ở trẻ em
Danh sách 21 biến số: id, tuoi, gioi_tinh, nl, pro_ts, pro_dv, lip_ts, lip_tv,
                   glucid, chat_xo, canxi, sat, kem, magie, phospho, iod,
                   vit_a, vit_b1, vit_b2, vit_pp, vit_c.
"""

import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')


def get_rda(age, gender, nutrient):
    """Tra cứu RDA theo Nhu cầu dinh dưỡng khuyến nghị cho người Việt Nam."""
    try:
        age_val = float(age)
        if np.isnan(age_val):
            age_grp = '3-5'
        elif age_val < 3:
            age_grp = '1-2'
        elif age_val < 6:
            age_grp = '3-5'
        elif age_val < 8:
            age_grp = '6-7'
        else:
            age_grp = '8-9'
    except (ValueError, TypeError):
        age_grp = '3-5'

    gen_str = 'Nam' if str(gender).strip() in ['1', '1.0', 'Nam', 'nam'] else 'Nu'

    # Từ điển RDA thu gọn (chỉ gồm các chất có trong danh sách 21 biến)
    rda_dict = {
        'nl': {
            'Nam': {'1-2': 1300, '3-5': 1600, '6-7': 1800, '8-9': 1800},
            'Nu': {'1-2': 1300, '3-5': 1600, '6-7': 1700, '8-9': 1700},
        },
        'pro_ts': {
            'Nam': {'1-2': 20, '3-5': 25, '6-7': 32, '8-9': 38},
            'Nu': {'1-2': 20, '3-5': 25, '6-7': 32, '8-9': 39},
        },
        'pro_dv': {
            'Nam': {'1-2': 10, '3-5': 15, '6-7': 16, '8-9': 19},
            'Nu': {'1-2': 10, '3-5': 15, '6-7': 16, '8-9': 20},
        },
        'lip_ts': {
            'Nam': {'1-2': 33, '3-5': 36, '6-7': 35, '8-9': 40},
            'Nu': {'1-2': 31, '3-5': 34, '6-7': 32, '8-9': 38},
        },
        'chat_xo': {
            'Nam': {'1-2': 8, '3-5': 8, '6-7': 10, '8-9': 10},
            'Nu': {'1-2': 8, '3-5': 8, '6-7': 10, '8-9': 10},
        },
        'canxi': {
            'Nam': {'1-2': 500, '3-5': 600, '6-7': 600, '8-9': 700},
            'Nu': {'1-2': 500, '3-5': 600, '6-7': 600, '8-9': 750},
        },
        'sat': {
            'Nam': {'1-2': 5.4, '3-5': 5.5, '6-7': 7.2, '8-9': 8.9},
            'Nu': {'1-2': 5.1, '3-5': 5.4, '6-7': 7.1, '8-9': 8.9},
        },
        'kem': {
            'Nam': {'1-2': 4.1, '3-5': 4.8, '6-7': 5.6, '8-9': 6.0},
            'Nu': {'1-2': 4.1, '3-5': 4.8, '6-7': 5.6, '8-9': 5.6},
        },
        'magie': {
            'Nam': {'1-2': 70, '3-5': 100, '6-7': 130, '8-9': 170},
            'Nu': {'1-2': 70, '3-5': 100, '6-7': 130, '8-9': 160},
        },
        'phospho': {
            'Nam': {'1-2': 500, '3-5': 700, '6-7': 900, '8-9': 1000},
            'Nu': {'1-2': 500, '3-5': 700, '6-7': 800, '8-9': 1000},
        },
        'iod': {
            'Nam': {'1-2': 90, '3-5': 90, '6-7': 90, '8-9': 120},
            'Nu': {'1-2': 90, '3-5': 90, '6-7': 90, '8-9': 120},
        },
        'vit_a': {
            'Nam': {'1-2': 400, '3-5': 500, '6-7': 450, '8-9': 500},
            'Nu': {'1-2': 350, '3-5': 400, '6-7': 400, '8-9': 500},
        },
        'vit_b1': {
            'Nam': {'1-2': 0.5, '3-5': 0.7, '6-7': 0.8, '8-9': 1.0},
            'Nu': {'1-2': 0.5, '3-5': 0.7, '6-7': 0.8, '8-9': 0.9},
        },
        'vit_b2': {
            'Nam': {'1-2': 0.6, '3-5': 0.8, '6-7': 0.9, '8-9': 1.1},
            'Nu': {'1-2': 0.5, '3-5': 0.8, '6-7': 0.9, '8-9': 1.0},
        },
        'vit_pp': {
            'Nam': {'1-2': 6, '3-5': 8, '6-7': 8, '8-9': 12},
            'Nu': {'1-2': 6, '3-5': 8, '6-7': 8, '8-9': 12},
        },
        'vit_c': {
            'Nam': {'1-2': 35, '3-5': 40, '6-7': 55, '8-9': 60},
            'Nu': {'1-2': 35, '3-5': 40, '6-7': 55, '8-9': 60},
        },
    }
    if nutrient in rda_dict:
        return rda_dict[nutrient][gen_str][age_grp]
    return np.nan


def format_mean_sd(data, is_pct=False):
    """Định dạng chuỗi kết quả Trung bình ± SD theo chuẩn y học."""
    data = pd.to_numeric(data, errors='coerce').dropna()
    if len(data) == 0:
        return '-'
    mean_val = np.mean(data)
    sd_val = np.std(data, ddof=1) if len(data) > 1 else 0.0

    if is_pct:
        return f'{mean_val:.1f} ± {sd_val:.1f}%'.replace('.', ',')
    return f'{mean_val:.2f} ± {sd_val:.2f}'.replace('.', ',')


def calc_met_status(grp_df, nut_name):
    """Tính số lượng và tỷ lệ % trẻ Đạt / Chưa đạt KNC."""
    pct_col_name = f'pct_{nut_name}'
    if pct_col_name not in grp_df.columns:
        return '-', '-'

    pct_col = pd.to_numeric(grp_df[pct_col_name], errors='coerce').dropna()
    n_sub = len(pct_col)
    if n_sub == 0:
        return '-', '-'

    not_met = (pct_col < 100).sum()
    met = (pct_col >= 100).sum()

    pct_not_met = (not_met / n_sub) * 100
    pct_met = (met / n_sub) * 100

    str_not_met = f'{not_met} ({pct_not_met:.1f}%)'.replace('.', ',')
    str_met = f'{met} ({pct_met:.1f}%)'.replace('.', ',')

    return str_not_met, str_met


if __name__ == '__main__':
    print('--------------------------------------------------')
    print('Đang đọc dữ liệu từ file CSV...')

    try:
        df = pd.read_csv('Data_diet_ASD.csv', encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv('Data_diet_ASD.csv', encoding='latin1')

    # Làm sạch tên cột
    df.columns = df.columns.str.strip()
    print(f'Đã tải thành công file với {len(df)} đối tượng và {len(df.columns)} cột.')

    # Chuẩn hóa giới tính
    if 'gioi_tinh' in df.columns:
        df['gioi_tinh_str'] = df['gioi_tinh'].apply(
            lambda x: 'Nam' if str(x).strip() in ['1', '1.0', 'Nam', 'nam'] else 'Nữ'
        )
    else:
        df['gioi_tinh_str'] = 'Chung'

    # Danh sách 18 chất dinh dưỡng cần phân tích
    nutrients = [
        'nl', 'pro_ts', 'pro_dv', 'lip_ts', 'lip_tv', 'glucid',
        'chat_xo', 'canxi', 'sat', 'kem', 'magie', 'phospho',
        'iod', 'vit_a', 'vit_b1', 'vit_b2', 'vit_pp', 'vit_c'
    ]

    # Danh sách các chất có định mức RDA để tra cứu (16 chất)
    rda_nutrients = [
        'nl', 'pro_ts', 'pro_dv', 'lip_ts', 'chat_xo', 'canxi',
        'sat', 'kem', 'magie', 'phospho', 'iod', 'vit_a',
        'vit_b1', 'vit_b2', 'vit_pp', 'vit_c'
    ]

    active_rda_nutrients = [
        nut for nut in rda_nutrients
        if nut in df.columns and 'tuoi' in df.columns and 'gioi_tinh' in df.columns
    ]

    # Tính % RDA cho từng chất của từng cá thể
    for nut in active_rda_nutrients:
        df[f'rda_{nut}'] = df.apply(
            lambda row: get_rda(row['tuoi'], row['gioi_tinh'], nut), axis=1
        )
        df[f'pct_{nut}'] = (
            pd.to_numeric(df[nut], errors='coerce') / df[f'rda_{nut}']
        ) * 100

    # Phân nhóm đối tượng
    total_df = df
    nam_df = df[df['gioi_tinh_str'] == 'Nam']
    nu_df = df[df['gioi_tinh_str'] == 'Nữ']

    n_total, n_nam, n_nu = len(total_df), len(nam_df), len(nu_df)

    # 1. TỔNG HỢP BẢNG 10 (Mức khẩu phần TB ± SD & % RDA)
    print('Đang xử lý Bảng 10...')
    bang_10_data = []

    for nut in nutrients:
        col_total = pd.to_numeric(total_df[nut], errors='coerce')
        col_nam = pd.to_numeric(nam_df[nut], errors='coerce')
        col_nu = pd.to_numeric(nu_df[nut], errors='coerce')

        mean_sd_total = format_mean_sd(col_total)
        mean_sd_nam = format_mean_sd(col_nam)
        mean_sd_nu = format_mean_sd(col_nu)

        pct_col = f'pct_{nut}'
        if pct_col in df.columns:
            pct_str_total = format_mean_sd(total_df[pct_col], is_pct=True)
            pct_str_nam = format_mean_sd(nam_df[pct_col], is_pct=True)
            pct_str_nu = format_mean_sd(nu_df[pct_col], is_pct=True)
        else:
            pct_str_total = pct_str_nam = pct_str_nu = '-'

        bang_10_data.append({
            'Các chất dinh dưỡng': nut.upper(),
            f'Chung (n={n_total}) (TB ± SD)': mean_sd_total,
            'Mức đáp ứng KNC (%) Chung': pct_str_total,
            f'Nam (n={n_nam}) (TB ± SD)': mean_sd_nam,
            'Mức đáp ứng KNC (%) Nam': pct_str_nam,
            f'Nữ (n={n_nu}) (TB ± SD)': mean_sd_nu,
            'Mức đáp ứng KNC (%) Nữ': pct_str_nu,
        })

    df_bang10 = pd.DataFrame(bang_10_data)

    # 2. TỔNG HỢP BẢNG 11 (Tỷ lệ Đạt / Chưa đạt KNC)
    print('Đang xử lý Bảng 11...')
    bang_11_data = []

    for nut in active_rda_nutrients:
        not_met_total, met_total = calc_met_status(total_df, nut)
        not_met_nam, met_nam = calc_met_status(nam_df, nut)
        not_met_nu, met_nu = calc_met_status(nu_df, nut)

        bang_11_data.append({
            'Các chất dinh dưỡng': nut.upper(),
            'Chung - Chưa đáp ứng KNC n(%)': not_met_total,
            'Chung - Đạt KNC n(%)': met_total,
            'Nam - Chưa đáp ứng KNC n(%)': not_met_nam,
            'Nam - Đạt KNC n(%)': met_nam,
            'Nữ - Chưa đáp ứng KNC n(%)': not_met_nu,
            'Nữ - Đạt KNC n(%)': met_nu,
        })

    df_bang11 = pd.DataFrame(bang_11_data)

    # XUẤT RA FILE EXCEL
    output_file = 'Bang_10_11_GioiTinh_Ket_qua.xlsx'
    with pd.ExcelWriter(output_file) as writer:
        df_bang10.to_excel(writer, sheet_name='Bảng 10 (TB & %Đáp ứng)', index=False)
        df_bang11.to_excel(writer, sheet_name='Bảng 11 (Tỷ lệ Đáp ứng)', index=False)

    print(f"\n🎉 Hoàn thành! Kết quả đã xuất ra file: '{output_file}'")
    print('--------------------------------------------------')