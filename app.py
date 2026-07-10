import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import re
import math
import json

st.set_page_config(
    page_title="Pakistan Neurogenomic Variant Repository (PNVR)",
    layout="wide",
    page_icon="🧬"
)

px.defaults.template = "plotly_white"

# ===================================================
# UNIQUE LOGO: Pakistan Neurogenomic Variant Repository (PNVR)
# ===================================================
LOGO_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 150" width="100%" height="100%">
  <defs>
    <!-- Glossy 3D badge background -->
    <linearGradient id="badgeGrad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%"  stop-color="#0f6286"/>
      <stop offset="45%" stop-color="#0a4f6c"/>
      <stop offset="100%" stop-color="#062f41"/>
    </linearGradient>
    <linearGradient id="badgeGloss" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%"  stop-color="#ffffff" stop-opacity="0.30"/>
      <stop offset="100%" stop-color="#ffffff" stop-opacity="0"/>
    </linearGradient>
    <!-- Realistic, shaded brain (light from top-left) -->
    <radialGradient id="brainGrad" cx="38%" cy="32%" r="78%">
      <stop offset="0%"  stop-color="#ffe2dc"/>
      <stop offset="42%" stop-color="#e7a7a3"/>
      <stop offset="78%" stop-color="#bd7077"/>
      <stop offset="100%" stop-color="#8f5260"/>
    </radialGradient>
    <linearGradient id="brainShade" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%"  stop-color="#ffffff" stop-opacity="0.45"/>
      <stop offset="55%" stop-color="#ffffff" stop-opacity="0"/>
      <stop offset="100%" stop-color="#3d1f2b" stop-opacity="0.38"/>
    </linearGradient>
    <!-- DNA strand gradients -->
    <linearGradient id="dnaTeal" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#8ff3df"/><stop offset="100%" stop-color="#15a48d"/>
    </linearGradient>
    <linearGradient id="dnaGold" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#ffe08a"/><stop offset="100%" stop-color="#ef9c1f"/>
    </linearGradient>
    <filter id="softShadow" x="-30%" y="-30%" width="160%" height="160%">
      <feDropShadow dx="1.4" dy="2.4" stdDeviation="2.6" flood-color="#001018" flood-opacity="0.45"/>
    </filter>
    <filter id="dropBadge" x="-12%" y="-12%" width="124%" height="124%">
      <feDropShadow dx="0" dy="3" stdDeviation="4" flood-color="#00000055"/>
    </filter>
  </defs>

  <!-- Badge -->
  <rect x="2" y="2" width="296" height="146" rx="20" ry="20" fill="url(#badgeGrad)" filter="url(#dropBadge)"/>
  <rect x="2" y="2" width="296" height="146" rx="20" ry="20" fill="none" stroke="#2fb6c9" stroke-opacity="0.35" stroke-width="1.5"/>
  <path d="M 14,4 H 286 a16,16 0 0 1 16,16 V 64 a150,40 0 0 1 -304,0 V 20 a16,16 0 0 1 16,-16 Z" fill="url(#badgeGloss)"/>

  <!-- ===== Brain (left), shaded for 3D ===== -->
  <g transform="translate(40,20)" filter="url(#softShadow)">
    <ellipse cx="34" cy="60" rx="36" ry="9" fill="#001018" fill-opacity="0.28"/>
    <!-- cerebellum -->
    <path d="M50,44 q16,-2 18,12 q-2,12 -16,12 q-12,-1 -12,-12 q0,-10 10,-12 Z" fill="url(#brainGrad)"/>
    <path d="M50,46 q12,0 14,9 M48,52 q12,1 16,7 M48,58 q10,2 14,5" fill="none" stroke="#7c4654" stroke-opacity="0.5" stroke-width="1"/>
    <!-- brainstem -->
    <path d="M40,56 q-2,12 4,18 q4,5 9,3" fill="none" stroke="#b5757c" stroke-width="5" stroke-linecap="round"/>
    <!-- cerebrum -->
    <path d="M8,30
             C2,18 12,6 26,8
             C30,0 44,0 48,8
             C56,2 70,8 68,20
             C76,24 74,38 64,42
             C66,52 54,58 44,54
             C40,62 24,62 20,52
             C8,54 0,42 8,30 Z" fill="url(#brainGrad)"/>
    <path d="M8,30
             C2,18 12,6 26,8
             C30,0 44,0 48,8
             C56,2 70,8 68,20
             C76,24 74,38 64,42
             C66,52 54,58 44,54
             C40,62 24,62 20,52
             C8,54 0,42 8,30 Z" fill="url(#brainShade)"/>
    <!-- gyri / sulci -->
    <g fill="none" stroke="#8a4f5b" stroke-opacity="0.55" stroke-width="1.2" stroke-linecap="round">
      <path d="M14,24 q10,-6 18,2 q8,6 16,-1 q9,-5 16,3"/>
      <path d="M10,34 q12,7 24,1 q12,-6 26,3"/>
      <path d="M16,44 q10,7 22,2 q11,-5 22,2"/>
      <path d="M30,12 q3,12 0,22 q-2,10 4,18"/>
      <path d="M48,10 q2,14 -2,24"/>
    </g>
    <!-- specular highlight -->
    <ellipse cx="24" cy="20" rx="11" ry="6" fill="#ffffff" fill-opacity="0.35"/>
  </g>

  <!-- ===== DNA double helix (right), 3D ribbons ===== -->
  <g transform="translate(196,18)">
    <!-- back strand (darker, lower opacity = depth) -->
    <path d="M6,4 C34,18 -10,32 18,46 C46,60 2,74 30,86"
          fill="none" stroke="url(#dnaGold)" stroke-width="6" stroke-linecap="round" opacity="0.55"/>
    <!-- rungs (base pairs) -->
    <g stroke-linecap="round">
      <line x1="9"  y1="10" x2="29" y2="10" stroke="#dff6ff" stroke-width="2.4" opacity="0.85"/>
      <line x1="3"  y1="22" x2="23" y2="22" stroke="#ffe9b8" stroke-width="2.4" opacity="0.85"/>
      <line x1="9"  y1="34" x2="29" y2="34" stroke="#dff6ff" stroke-width="2.4" opacity="0.85"/>
      <line x1="3"  y1="46" x2="23" y2="46" stroke="#ffe9b8" stroke-width="2.4" opacity="0.85"/>
      <line x1="9"  y1="58" x2="29" y2="58" stroke="#dff6ff" stroke-width="2.4" opacity="0.85"/>
      <line x1="3"  y1="70" x2="23" y2="70" stroke="#ffe9b8" stroke-width="2.4" opacity="0.85"/>
      <line x1="9"  y1="82" x2="29" y2="82" stroke="#dff6ff" stroke-width="2.4" opacity="0.85"/>
    </g>
    <!-- front strand (brighter, on top) -->
    <path d="M30,4 C2,18 46,32 18,46 C-10,60 34,74 6,86"
          fill="none" stroke="url(#dnaTeal)" stroke-width="6" stroke-linecap="round"/>
    <!-- strand highlight -->
    <path d="M30,4 C2,18 46,32 18,46 C-10,60 34,74 6,86"
          fill="none" stroke="#ffffff" stroke-width="1.6" stroke-linecap="round" opacity="0.4"/>
    <!-- mutation spark -->
    <g transform="translate(40,8)">
      <polygon points="0,-7 2,-2 7,-2 3,1 4,6 0,3 -4,6 -3,1 -7,-2 -2,-2" fill="#FFD54A" stroke="#fff" stroke-width="0.6"/>
    </g>
  </g>

  <!-- ===== Title ===== -->
  <text x="150" y="120" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-weight="bold" font-size="10.5" fill="#FFFFFF" letter-spacing="0.6">PAKISTAN NEUROGENOMIC VARIANT REPOSITORY</text>
  <text x="118" y="137" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-weight="bold" font-size="12" fill="#FFD966" letter-spacing="2">PNVR</text>
  <text x="200" y="137" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="8" fill="#CDEAF2" letter-spacing="0.4">Clinical Genomics Curation</text>
</svg>
"""

# ===================================================
# DASHBOARD LOGO (embedded PNVR brand image, base64 data URI)
# ===================================================
LOGO_IMG_URI = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD//gAQTGF2YzYwLjMxLjEwMgD/2wBDAAMCAgICAgMCAgIDAwMDBAYEBAQEBAgGBgUGCQgKCgkICQkKDA8MCgsOCwkJDRENDg8QEBEQCgwSExIQEw8QEBD/2wBDAQMDAwQDBAgEBAgQCwkLEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBD/wAARCAEtAfwDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD9UKKKgF3G03kKQZByV7gZxz+tFhNpbk9FFFAxN3+zS0A9OKhafZywO3nLdhgA80bCuTUVV/tGzJx9oT86eLy2P/Ldfzpcy7k+0h3J6Kh+12//AD2T86T7da/8/EX/AH0P8aOZD54k9FQfbrP/AJ+Yv++x/jS/bbT/AJ+F/Oi6Fzx7k1Ju/wBmoDf2o6TqT7EH+VPW4jZcgE4ODTGpRezJaKYsgfBXofXj9Kd82aOlyhaKKKACikY4x781Wa/gUlTNGMHHL9wcelCTexLko7lqiqwvoG6SJ0z97I9+RxUwkU9xycU7NbjTT2H0UUUhhRRRQAUUUUAFJu/2aY0wQbmyFGckj9fpTftlvnG8Z+tNJvYlzinZsmoqISr/AHv5U37TuPyKWGcDB6+p59DRZ7hzp6XJ6KrG9iDBTKAT2OO44oOoWw6yrRZidSEXZvUs0VVW+jc7FIJJAyGB65xx+HSrIOfpSatuUpJq6FooooGFFFFABRRRQAUUUMMdDQAUVWN9CH2Bs8kE44BH8+SKct5bk4Eyk01GT6EOpBOzepPRUH2yD5f3q/NjbznOeBU2fehq25SknsLSbv8AZqFryBWCtIo3MFHOOef8KcLqA9JFP40+Vi5kSbv9mlqP7TB/z0X86PPi/vilYOaPckoqP7RD/wA9FpfPg/56D86Vg5o9x9FRieI9H/rQZlxnr7DqPw60agpxezJKTd/s0wTLkAhgScAEd6kyB70J3K3CiiigAooooAKKKKADNfOfh55pP24vFdu9xIYYfAunMsZc7cm6kyQOgzX0Wa+c/DPzftxeMiO3gPTB/wCTT00cmJdnBd2fRtFJu9qWkdYVxXxhynwt8XNG7o66JfEOrEFSIWwQe1drXG/F7B+FvjDP/QDv/wD0Q9KWxz4n+DJrsfkA2o6k5+bU7884/wCPqT0HvTRqmqjprGof+Bcn+NV2G1QS3r/IVD5vNeZdn4/OtUv8TNH+1dU/6C2of+Bcn/xVP/tjUv8AoLX/AP4FSf41mrJkE88YBPb16fkPxoEsR6kD/gYouyHOa3b+80f7X1H/AKCeof8AgZL/APFUn9q6h/0FtQ/8C5P/AIqqX0X/AD+dJuX0b/vmjXuL2s+7/EtSa5ryDMWv6rGf9m+lH/s1dJ4d+NPxf8KSrLoXxL8RW+z+CS8adD/wCTcv6VxrE/xI2e/TimFiP4fb8fSq52upcMVXi/dk/vPrv4Wf8FBvGWjyw6f8UdBg1uzJAa/sgsNygz1K/wCrf8NtfbHw7+K3gf4o6MmueCtch1G3OBKinbLA5/gkjOGRvYjHoSOa/GtpCMYkfI77jXR+AfiP4s+GPiO28T+DNUksb6DGQpLRToOqSp0dD6VtCq09T6DL+JsThmo4j3o9+qP2f3nrt4PfPb1p9eQ/s/ftA+Hfjp4VTVdPVLPVrPZHqenF8vBIQcMvdo3xlW9iD0r13IHOa60+bVH6Fh8RTxVNVKTumRzf6th/skfpX4j6z4g10apqAGu6mB9pdeLyT++f9qv24mHysfYmvwz1l86rqIx/y9Of/HzXVh+tz5Li2coqlbz/AEL9v4x8X2NzBqGm+K9ZgvLV1kikF9KSjDp/Fj8K/VP9lv48WHxw+HNpqrvHHrlhi21e3U8pcDH7wL1COPmX6kdq/JBGCv8AKxHvXp37PPxs1H4EfEGz8XQLJPpc5+zavYqf9faluWA/voRuX3yOhIrWrTVSN0eFkubTwOJSqP3XufsgMntS7fesfw74m0nxVolh4h0C/jvNO1OBLi2uIzlZI2GQR+fTseD0rZBz3rhtY/T4yU0pR2Ym33pKfTdvvQUJRS7fekPHWgDlPiUZI/h94llikaNk0i9ZSpwVPktyD1FfjVD4i8SbAzeJ9YY9f+QhN/8AFV+yvxM5+HfifP8A0B73/wBEPX4tx4W3V2OMqMj8K9TL0pJ3PgOMKkoVIcr38zR/4SfxIP8AmZNX/wDA+b/4qv09/YkvLjU/2dPD9zfXMs8zz34aSWQuxxcSAcnmvyqZgvU5r9Tf2EQrfs2+HSGzi41Hn/t6kqsfFKF0cvCdWpPGyUm/h/VHgX/BQzw3qnhnxxonjPTbu+isNdszZS+XcSIkdzCOOAcDMbA/8ANfIH9t6vuI/tnUv/AyX/4qv1T/AGvPhp/wtL4H67pttC0mp6Uv9q2AU5bz4FZmUeu5N6Y9TX5NJLvAkUfK6nBz3xn8uR+lPBuM4arVHPxNQq4TGucW7S1O9+E3xb8SfDL4g6D4x/tvUZrWyvR9qhmupHR7YjZINpJBOwsenBxiv2NsNTttSsra/spVlguY0ljdDwyMAVb6HNfhmW3naQevr/n0FfqP+wx8Sx49+CNjo15c79S8KSNo9yWOWMS8wOfYoQvPVkas8dRSjzo9HhDHtVJYWb31R9JUUDnpRXmH6AFFFFABRRRQAVE0vzEbSeSM/hn/AB/KpDnoDiuF+MnxCsvhb8M/EHju+cY0uyeSBCf9ZO3yxJ9S5UfiacVzNJGdWrGjTdST0R+d37aHxc1fxf8AHLV9L0jW72LSPDWzSYkgunRHnQFpmIUjJEh25/2a8LGua8hJXxDq5P8A1/zf/FVn3V3dX11Pe3lw8s91O9zNIxyXkYksx+pJqFZXzgmvpKdCMIqLPxfG4+ticRKopNa6ansv7N+h+JPiX8YPDHhNdd1iS1a6XUL4i/m+W1h+dj97jPC/VhX617yVx7V8Sf8ABOL4b/Z9D174q6hAVk1CVdK05yMnyYypmYexkwP+2Zr7Zk4BIrxsbJSq2SP0bhrDTo4Hnm3eWp+MXiLxFrR8Q6wG17VSFv5go+3zAACRv9qsseJvEA6eItX/APA+b/4qmeIZAPEOsH/qIT/+jGrL3e9ezGEWr2PzOvXqKpJJv7za/wCEm8Q/9DHq/wD4MJv/AIqj/hKPEn/Qzax/4MJv/iqxdylVKhmJ/ujI/wDr/wD66f8AOOoP6VXJ5GLrVVu395q/8JR4l/6GPWf/AAYTf/FU7/hKPEx/5mbWR/3EZ/8A4qscCQ/wuKJXMY5X82o5EHta3d/ebCeMPFcf3PFmuj6apP8A/F1vaF8cvjF4Yk8zQvif4itT/ca/eZP++ZCy/pXBJJv+6uR67hTt6gZyOu3qP8aTpwl0NY4qvB3U39//AAT60+Gf/BQj4meHHht/iHo9p4m0/IVriFRbXKA+6go302g+9fbfwn+O/wAOfjJpZ1DwRrguJ4lVriwlHl3Vvn+/Gecf7QyPfPFfjhlA2/7zepIz/KtHw94q8R+DtctPEvhTWLjS9TsXDQXFudu0dwR0ZT3BBDdwa5a2BhUV46M+jy7ijE4VqFf3ofj95+3yyqwyoJBxUu33r54/ZT/adsPjvoR03V0isfF+lxj+0LNWAS4TPE8P+ycjI/hJHXrX0OCa8acHTk4yP0jC4qnjKSq0ndMSiiioOgKKKKAEPb6185eGCR+3J40A/wChD0w/+TTV9G9TXzn4aAH7c3jIDv4C03/0qanFbnHit6fqfRQ56U+mLT6R2BXFfGM4+FHjI+mhX/8A6Jeu1rifjKMfCfxn/wBgDUP/AEQ9J7HPif4EvRn44+bvVeMf/qqNiRg0yJgqpk5//VUYd321wWPxya98+qf2Yf2TPCXx6+H8/i/XvE2sabPb6pNZeVZiLayokbBsspOfnP5V7KP+CbPwzH/M/eKsfW3/APjdX/8AgnL/AMkT1D28R3Of+/UIr6www712Rpx5T9FyvJ8HWwsKlSF2fIA/4Jq/DFfu+PPFI/8AAf8A+N0w/wDBNb4aN0+IPigf+A3/AMbr7C2n1oyPf86PZR7Hof2HgP8An2j4zu/+CaPgAjFj8SPEMbdjLBbyD9FWvKviP+wB8SvClpLqHgrXLTxRFFljb+X9muSPRVYlGPtuz6A8Z/SHj0pnlLnOT19e1DpRZz4nhzAV42UbPyPxE1HT7vS7u6stQtJraezfy7iGZCksTg4Ksh5VgQQRyc/nVJyO+ME5I7E1+jf7Z37Nem/EPwve/EXwpYFfFeiQGWQQgBr+3QZaNgPvOqqSh64G3vivzb81plVkTG4leTxuxkD9R+lYey5GfAZlldTLqvs5ap7He/CH4wa98GfiJpvjvTJXlghZbfUbQHi6syBvQ/7RABXPAZBX7AeG/EuneKNCsdf0m4W4s9Rt0uoJU6SRuAVI/AjNfh+p4YMWO4EHpyD1r9I/+CefjuTxN8ILrwleXDyT+E757RCzZP2WX95Hj6FpFH+6PSrptp6H0HCuMdOo8LLZ6ryPq+Vsq/0P8q/C7V/+QxqOP+fqT/0M1+6EhO1/p/Svwv1XA1nUgT/y9y/+hGvSwy5mzo4uXu0vn+hAAoTJppkA6DHzZJHWonkbOcUxpB7V1pdj4bld7M+yf2C/2jh4S1mP4LeMtQP9kanOZNBuJn4t7lufIyf4JCTjGMP/ALxx+jSXKsRswckd+mf/AK2fyr8GjM6DdHI8bhg6OrYZGBBBBHQ5APHcZr9VP2Lf2kI/jb4GTw/4hulPi/w3FHDfqTzdw9I7pR6Ho391h2DDPJiKdtYn3XDOauS+q1n6f5H01RTcnOKXJB5rlPtBaY3OaXd7U1m68UAcp8TD/wAW68Tn/qDXv/oh6/FEygwKc9hX7XfE1h/wrnxRx/zBr0/+QHr8REmzCqH0r1Mu2kfBcXxvVp/MklnwcZFfqp+wMwP7M3h4/wDT1qA/8mpK/KOaU4+6M1+rP/BP91b9mTw6cdbvUR/5NSVeP+BepzcJq2Mb/u/qj6KlgV0MfZgQM88V+O37R3wxk+Evxl8SeGCpjszcnUdMzwrWsxLIBgdiWT6x1+x3XrXxL/wUj+Gjaj4S0T4r6dAxn0O4On6gw4/0aY/u2Pss3H/bWuLA1OSpboz6DijA/WMJ7Rbw/I/P5pFznivo79g74qDwF8arfw7fT7dM8Yxf2ZJuf5Vu1+a3Y+uclOf+elfNLMDnB6U+31S/0y9ttQsLiSG6s50uIHRsGOVCCrD3BA/KvaqQVSLTPz7A15YTEQqx6M/eBJOm3nP+OKfXn3wR+I1r8Vfhh4d8eWZVf7Vs0klQc7JxlZU/4C6sPyr0HB55r5yS5W4s/Y6VWNWCnHZhRS7fekpGgUUUUAISMbq+D/8AgpR8TfLj8PfCTT7gfvm/tnUwrfwjKwKcdiRK3/AVr7lvb+3srWa6uZBHDbo0srtwERRknP4V+MPxy+JkvxY+KXiLx3udoNSvWSzV1+5aRfLEvt8gDfVjXfl9Pnqc3Y+Y4oxio4T2Ud5bnGecPm461a03Tb/W9VsNH0m3a4vdRuUtYIVBy0jkKg6dSWH41mBs/nX05+wP8Mm8c/GqPxPfWxfTvB8H2xyVyhuXysA+ud7f8Ar2a8lCDkz87y/BvF4mNJdz9E/hP4Csvht8PtB8EaaF8rR7GO3d1GPMkxmVvqzbj+NdfL9xvanJEAvB/wA5zSSr8hOfrXy/M5Suz9kjTVGjyR2SPxB8RS/8T/VxnrfzH/yI1ZqzKn3jipvEc2PEGrYOf9NlH/kVqzTKD1PevqIr3Ufi1aneoz6X/ZV/Zp8OftGWviOfXPFGpaV/Yb2yRrapG2/zRJk/NnGNnb1r6Ej/AOCafw+T/moWvn620H+Fc1/wTFlZrDx/3H2jTh+lxX3aMDivHxVepCo0mfoWS5NgsRg4VKkLv5/5nxz/AMO2fAB6fEDxB/4DQf4VBJ/wTO+H8rZk+I3iH/wHg/wr7Lx/nFG36flXP9aq9z1f9X8t/wCfS+9/5nxZJ/wTH8Bqv+jfEzX4z/tWsL/yFeZ/Ef8A4J3/ABF8M2Euo+APE9n4nij+b7LLELa5Zf8AYyWjY/Vlr9HjyMHmozChBDZcE8g9MfSqhjKsXe5jiOGsBWhZRsz8ONW0/U9D1G70nWbCexvLJ/Knt7lTHLHJ/dZDyD055HI65FUt+/OMV+m/7YX7L9h8XPCd14w8OWezxpo8DSW0kCBGv4kGTbyEfeYgnyyehwO5r8wfOYAExbMnYQeoPp+WK9rC11XifAZpk88tquG6ex2fw8+I+tfDPxZpfjTw5cvHqGlTLMgZvllUABonx1VkyvsCMdBX7D/D34g6L8SPBejeN/D8oay1m2S4jUt8yMeGjP8AtKwYH3U1+IW85znlec1+h3/BNXx9JqXgnxJ8PLudmbQb9L61BOSILjOQB2AdWP1auXM6KlD2i3R7HCOKlh67wjfuy19D7XooorxT9GCiiigBB1r5z8Ocft0eLx/f8A6f/wClTV9GDrXzn4e/5Pp8Wj/qQLD/ANKjVR6nHit6f+I+il7U+mL2p9SdgVxHxnz/AMKm8aH00DUf/RDV29cP8aDj4R+Nif8AoX9R/wDSd6HqjnxS/cS9GfjB5u1U49f5Uiz/AHaqs4CrznrUaud230rm5T8jnE/TT/gnGN3wS1E56+I7n/0VDX1lXyT/AME3HB+B+pEcj/hJLn/0TDX1t2zXRHZH6llGmDp+iCmU7d7U2mekFLSUUAQyQg5JbIIKn6HoPwzX4zftD+Frf4ffHPxr4TsrZorOz1UzW0YGAkU37xQPorgf8Br9na/I/wDblnt3/af8YGBslE09HPuLVc/lwPxpNJo+Y4opwlhoye6eh4iJi3OOa+wP+CautS2XxP8AFfh/zcxalosd1tJ/jhmA6f7shH4V8bo4Vi2c19W/8E5/Mm+P926ltsXhq63Ef9doQM/mKxirM+Uyi8MfT5e5+mzcq+f7v9K/CnXGxreqYPIvJv8A0M1+65U4IJ6pX4T6+jJr2q7jn/TZf/Rhr1MGrtn0PFSvGnfz/QrcvxTJ4JYoYZ2SQJM7orlGCEqRuwSOcbhn0JGetQmXaNobn1r64+AXwMtfj9+yJ4g0iDy08QaT4kurvRrmRgMTfZ7fMRP/ADzkA2n0O1zkriuuo1T3Pk8JgZ4ubjHc+QzMD1UV1Xww+KniT4P+PNK+IHheXF3p8n7yFmxHdQkAPA+OSHXI+oBHIBrkbu01DTby803VbKWyvLCZrW6t5hteGZTgqw6ghht/H8ahd2cqxwe9NR5kZxUsPVUlpJfgfuZ8Kvij4c+L3gTSfH3hSfzLDVYtwViC8MgOHifHR1YMpHt7gntM+3Ffkx+w9+0g3wV8cr4Q8TX3l+DfE9wkcrOx22F3wsc+eykbUftja38Br9YEuVk2lWUhgGyDkYPIOfTg15dan7KR+nZVj1j6Ck/iW5NmmHmn0ysj0zlPifx8N/FRHbQ73/0Q9fh/G48tfpiv3A+J5/4tt4rH/UDvv/RD1+GyPhVGe+a9XLtpHxHFS5qkPRkspycCv1b/AOCfRP8AwzF4d/6/9R/9KpK/KAy85xX6u/8ABPZt/wCzF4dOP+X/AFL/ANKpKvMFamvU5OF1/tb9P1R9K7vauR+KXgWx+JPw+8QeBdRUGDWrCW13MoIjcr8j/UPtb/gIrraGQt1Xk148dGmff1KaqQcJbNH4P6ppuoaHqN/omqwGC90y6ls7iJsgiVGZWXp2ZSD9Kos5655r6b/4KEfC0eBvjY3i+xttmmeNIBfAhMKl3HhJwPc/I/uZCa+XjLnrX01CfPTUkfkeNwzwteVLsz79/wCCZfxQzbeJPg/qVwS1sw1vS1LcmNyFmUfRthx/tmvv0HIzmvxE+APxJk+E3xk8LePDMVtrG9FvqIU/etJfklz7BW3f8BFfthaXiXNvFPC6ukqhlZTkEHoR9RXkZhS9nUutmfecNYr2+EVOW8X+Bdpu33p1FcB9GMPHWkZgKU89aheQL17HFAHzd+3f8Vj8OPgRqenWly0eqeK5P7GtArYby3yZ5B7CMMM+rrX5SiQJgrggYwK+lP8AgoJ8V/8AhO/jfN4W0+4Eul+CYRpyBTw14+HuG/D5E+qGvmLeQ1fRYCj7Okn1Z+aZ/iHicU7bIvLKhOA3IJzzgYxwR7fWv1a/Ya+GA+HnwP07UtRtBDqviphq1yCoyI2AFuh47IA31dq/NX4F/Dib4ufFzwx4BhVnh1K7SS+I6pZxgvKx+qKw/wB4r61+2VlYW9laxWlvGI4YEWONF4CKowAPQAVy5nV5UoI9ThXApSliWvT1LAOcCop2CqQR2qUKR3qG6+430P8AWvGjufaVF7rPwr15i2u6o+c5vJT/AORD/iaocZ6981PrUm3WdRYnrdS/+jKob/evr4r3Ufj1SK52z9A/+CXw3ab8QMHpdad/6DNX3hXwX/wS3lDab8Qva603/wBAmr7z3e1fOY3+Kz9MyGNsDAX8T+dJu+v50tMrkPYCiiigCKSI4YeYec/yxX44/tT+FLPwJ+0F4z8OafB5FoNR+2W6KuAq3EYmwB2GXI9sCv2RkHyjmvyI/bpv4bn9qHxcIZS5tksbdj/tLaoT+PI/KvRyxv2rR8zxPCLw0ZPdPQ8O84Dtmvq//gnDrE1h8eL/AErcQmq6BMGHYtFIjA/ln86+RRIRX1J/wTtWW6/aOt5FO4W+g3sjnH3QTGoP5mvVxatQkfJZRGSx9JrufqpRTQ2e1Or5k/VLWCiiigQg6185eHW/4zs8XLj/AJkCw/8ASo19H7cV83eHf+T7vF3/AGINh/6VGrh1OPFbw/xH0avan0xe1PqDsCuF+N7bfg943YdvDuo/+k713VcH8c32fBvxy5HTw5qJ/wDJd6DDE/wZejPxMEuFXj1phuNvemE/KmfeoHXc2FbNRyn5TbXU/Tz/AIJnymT4G6oc8DxLcf8AoiCvsDtivjf/AIJjEp8C9WVsFv8AhJrgcMP+feD3r7Hyc5x+oq7aH6XlP+5015CbT60lO3f7J/OmkjtQejcKTd7UpP8AssP8KrT3aW/+tUryeTwMAdee3+c0A3Yh1TVrLR9Ou9V1K4W2tbGF7ieaQ4VIlBLOT6BQSfYV+JPxY8ft8TPiT4n8d7WUa3q01xCp/hgyViH12hR9RX1Z+3J+2Pp3iOzvPgp8KdSFzaNII/EGrQSbomAPNpCw+9yBvYcEAL3OfhzzFwBt4HHX/Peq5bKx8NxDjo4mSo03dItb85we9fa//BMHw3dT+L/G/jOVT5NrY22lxOejNI7SMB9PLT8xXxJCC/IA278bicD3z6cA81+uP7FHwmuPhb8CdJh1O2MOq6+x1i+VlG6NpQvlIf8AdiCZHruqZR0uc3D2H9ri1NbJfifQZ6H6Yr8G/Elww1/Vw3OL2Yf+RGr949w549a/BHxNKP8AhI9a5/5iE/8A6Mau/AK7Z7PEsFKNNPz/AEKhlz1Nfph/wTHVJPgbrikBgPEs4wef+WFt/hX5hiUnPNfpv/wS3Yt8EvEO7nHiebH/AIDW9bYzSnc8vh+DWMv5M4r/AIKG/s0eZHN8fvA9g0ksDCPxLbRLgvEAEW8A77chJD6EE9Gz8BxuCCwJ2sflPsRnr+X51++l9pdlqVncWGoW0Vza3UbxTwyoGSRHBDKwPUEEgjvmvyA/bA/ZyuP2f/H0selJM/hHXpGu9HmfJERyN9q5P8aZ4/vIynqGxlgq3MuSW508Q5Z7JqvTWj3PCFZcFW+YEYbPbp/gK/TP9gD9ppfHGhRfBvxtqO/xFocAbSrmWTLahZKuQpJ5aSMcHuV2nkqTX5ibzzz161reG/Fmv+DvEGm+KPDepvY6ppFwt1ZzoeYpFOQD6ryQR0IJB4JB66lBVY8p5GXYueBrKcXo9/Q/fUTKcYHXPU+lLXkn7N/xy0P49/DXT/GmmmOG9yLbVbRTk2l6oXcgB52EHepPVWHcHHrmOM57V40ouLaZ+kUqsasFOOzOS+KBx8NfFn/YDvv/AEnevwuVx5YPpX7ofFI5+G3iz/sB3/8A6IevwnWb5duK9PK1dSPkOJo3nB+pczvOCcV+sP8AwT0Yf8MweHgO2oan/wClUlfkqkuPmNfrJ/wTtfP7L/h9uudQ1PP/AIFSVrmK/dr1ObhqNsXL0/VH04OelP464/WmLTq8U+8Pmn9vL4Xf8LE+Amp6hZ2zTap4Ub+2bXYMt5SZWdf+/O9sf3kWvyXSQyIHUe3X6V++moabbalaTWN1CstvcRtFLG3R0YEMD7EE1+H/AMa/hnd/CH4o+J/h/OrBNI1BxatIOXtnw8De+Y2Q59c17WWVLp0+p8ZxNg0qkcQlvp/kcjGUZzG4G1hsP5Y/rX67/sR/FQfFL4CaI9zcCXU/D3/ElvuctuhwI3P+9HsOfc1+PjSnOe/Svrf/AIJtfFlfCHxkvPh/qV0I9O8aWgWBWbgX0AZ0x/vL5q47nYPSujH0faUm+qPPyGu8Ni7X0loz9Usik3e1Rh2yBjn60/NfOn6GFcF8ZPiJY/Cv4aeI/iBfEbNFsJZ0Q/8ALWc/LEn/AAJyo/4EK7stjtXwH/wU++Ky21h4d+C+mXGHvHGt6sFbpEpKQIfYtvb/AIAtb4el7WrGJw5jX+rYeUz4E1DUb3Vb+71bU7qS4vdQupLu5lY5LyuxYsfckk/jVYygHJFRiT5j6cfpUllp99rF9Z6XpkLXF1fzrbQRIPmeR2VVXHqSyge+a+sSUY+SPzHllWmu7Pvz/gmH8L2e28RfGfVLQK87DRNJdlyGjBV7hx7E+Wue2Hr9BhxxXnXwS+G1l8Jfhf4b+HlioP8AYtikc7qBiS5fmVz67nZ2PtXotfJYmq61VyZ+m5bh1hcNGml6hVe6/wBW3+6f5GrFVrs4jb/cb+RrCO52z+Fn4O6y3/E4vx/09S/+h1ntOF681Y1efOsX5PT7XJ/6Eaz3YHoK+y05Ufk84rnZ+hP/AASuJfS/iMf+nrTP/Rc1fe456V8Df8EqGA0j4i5/5+9N/wDRc1feqzLtUgHnkdOhr5nG39s0fouTtRwcNdCem7T600zAdQf0pi3Sv91H/KuQ9Pmj3JKGIHSoXuVj5dSFAyWPAGPWvKfip+1N8Dvg/ZSz+MvH2mi6iB/4l1lKLm8dgPuiNM7T7ttHqRVRhKbtFEVK9Okrzkkdn4++IHh74d+ENX8aeKLoWum6LbPc3Ejf7OQEX1djgKvUllHcGvxE8d+NtR8f+Nde8cauR9r17U5r51BJEYkOVUZ7AED8K9V/ai/a38V/tF6iNPt4W0Pwdp8qy2elrJukuZMZWa4YABmweP4V6DJ+avBjMWbeQD81fQ5fg3h4889z4fOcx+u1FTh8MS1v9819zf8ABLvwpLd+KPG3j6WLENjZ22kQSYJBlkPmOM+ypH/30K+EUdsxLGhkaVtiKvLM3pt65ORgdzgda/ZT9j/4Ov8ABb4JaH4cv4TFq9/nVdW9VuJNvyH/AHU2IfdCaWZ1FTpcvVlZBhXUxKqNaJfie5DjpT6aFJ706vnT7xu4UUUUCHGvm3QOf27/ABaP+pAsP/So19I7vavm7w7/AMn3+Ln/AOqf2H/pUauHU5MTvD1Po1e1Ppi9qdu9qg6xa4H47/L8GPHZPbw1qP8A6TvXfVwPx4H/ABZXx5/2LWo/+k8lBz4j+DL0Z+ILSZVO3X+lRO+D8p5qF58xoQKiEpHrWvKj8vcbvU+kf2ef21/Ef7Ovgq68G6R4E03WornUZNQNxPdvGwdkRMYVSMfIO/evTv8Ah618Qv8Aojehf+DOX/4ivhwup6n2pNy+1PkVj0qGY4qhBU4z0R9xN/wVZ+IA6fBzQ/8AwaTf/EVah/4Ks+OD974Q6Iv/AHE5v/iK+E9yHv8ArSFwfX86fIjb+18Z0mz7k1D/AIKn/EqRGWy+F/huFivyvJezyAfgAteGfFn9sb46/GO0uNG1/wAVf2Zo1wCsum6PF9mhkU9VdgTK6/7LOR7V4bvTGMc+tIGXOCMgU1FI56+PxNVctSehbUhYliQ4RPup0A6Z/kKVrkLgjBBP5Djn6VXXL8jBA5Jz0HXjIGfTNe2/s1/sj/EX9oXWob2S2n0XwbFN/pesSxczBW+aO3B4kbAxn7i9ySMFySRz0MLLFVPZwV2dd+xJ+z1e/G3x9F4g1uwb/hDvDc6zX0rodt3ODuS1Bx8x6M47LxnLiv1tgtViRRgKE6AcDv2/GuZ+G/w28JfC/wAJ6f4L8G6WljpmnRhYkQ/NIedzuf4nYkkk5JzzXW7vaueTbZ97luXQy+lyrd7kD/IMmvwH8SSbvEOstnrfzn/x9q/fmVTtxn/OK/n716YnXtWJH/L7N/6G1ehlyu2eZxCtIfP9CuGA71+oH/BLbA+CPiI9z4nk/D/Rbevy68w+lfp9/wAEspM/BPxKueniiT/0lt66McrUrnn5CrYz5H2vgbc+tea/HX4LeG/jj8OtR8A+IoyguwZbO8C5ks7pQfLmUdSQTyuRuUspxxXpXbFJsXbtxwe1eLF8ruj7OrTjWg6c1oz8C/HvgvxB8NvGWs+BfFdutrqmizmCVM8ScjDx8fMpUq6nupHfpg7un90dK/Uf9v79lofFbwo/xR8G2DN4u8NW7GaKFedSsUyzR8fekQBnj743L3WvyyWQsok24jJ25BH3u/HsD+lfQ4Wqq0fNH57mOBlhKrjbR7Hun7J/7Rd9+zx8SodYumml8MauyWmu2qHdiInIuEHd42JbHdSwJO4Y/ZTSNbsdb0y11bSrqK6tL2BbiCeFg8csTLkOpHUHI/Cv59lfa+8NnPBxxx6fTjpX35/wTq/agFvLH+z345vh5UrM/hi6nYghj8z2hJHHGWjH1XptA58bh7pVInq5Jj3SfsJvQ+8viihHw38Wc/8AMDv/AP0Q9fg6HAUc+tfvD8Tn8z4ceLAuCf7DvgR/dJt3PP6/nX4LBwVXPvTypfEzTiFKUoehYaT3r9bv+Cc5DfsueHj/ANRDU/8A0pevyIeQjpX65f8ABOElv2WfD7f9RHU//SlqrMv4a9Tn4fjbEt+X6o+pFU8c0tC9qK8Q+0DHNfnl/wAFQvhUou/Dnxl020G2T/iRamyDHzAs9u59z+9T8EHpX6G15p+0L8LofjB8H/FPgCVAZ9RsnayY/wDLO8Q74W9v3gTJ9GbtXRhavsaqkefmWG+tYaVPqfh6Tndk9BmrmheKtX8I+INN8U6LcNDf6NdxX9q4JyJY3Dgn24H171nzrcWs0lrdRGGeB2gljbgo6nBB/EN+VQM3avq2lKHqfncL0pp9UfvR8LviBpXxQ8A6B490ZwbTXLOG7Reu0uvKfVWyp91Ndlu9q+Cf+CWvxXOseB/EHwg1GcG58NXQ1GwBOSbK4JJUeyyhv+/i196ZG3NfI16XsqjifpWCrrEUIzRXur2C0gluLqVYooUaV5GPCooOST2xivxD/aH+Kkvxh+MHifx+JS9rqF8YrAMfu2UI8uED6qA2PUk1+k37ffxc/wCFW/s/6xa2F2ItX8WMdCsgr4YCUE3Dj02xBxnsxXHWvyHVwq4xkccHHGOn8hXqZTR3qHzvEGIcmqC2LXmZXrX1L/wT0+FX/Cw/jrD4kvrYyaX4JtxqLllypujlbdSfUMXf/tnXymkke4kttC/Mdw/hAz/jX65/8E//AITn4dfASw1e+sjBq3jOQ61ch1wVgYAW6e37sBue8hruzCs6VHTdnmZLhPbYlX2R9ORQ8AHvzU9JgA5pa+WT6s+/7IKqXn+pY99j/wAjVuqt4P3L/wC4/wDI1UdyZ/Cz8CNWcf2vfjPH2t/5moVb2zTdWcf2xfc/8vb/AMzTI7gL2zX2kV7p+Y1Yrmufop/wSwBXQ/iMQR/x+acP/Ic1cX/wU98SavYfE7wfY6brGo2kcmgSSPFb3Ekas/2lgCSpHOVPJ7AV1/8AwSqlaXQ/iPgf8v2nf+i5q9g/bB/ZWtv2h/B/2rSmWz8Y6HC/9lXckm2OeMtl7aX/AGXIGG6oxGSQWz4PtIUsa5T1Pq4UatbK4xhoz8oH8WeJn+74n1gf9xCb/wCKqWPxh4oi6eJtY/8AA+b/AOKrM1zQNa8MavfeH/EGnz6bqemzPbXVrOuJIpk4ZGHY9T6HGQSDVBpiMY7/AKV9AqcZLmSVj5d86fK27m5ceKNeuVCS63qTqQQQ15Ic569WrJMoGdiqCTktjLZ/GqzSe9PgxIQBPErHnDNg4xn09MHp0NO0Y6rQLSemv3kpdSSxZtx59qcrgtghguSN3TJAyMA9QfXtW/4L+GPxC+JeorpPw/8ACeo65dkgMtrCdkfu7n5UX/aYge9fev7Mf/BOGy8P3Np4y+PN1b6rqEDCW30KFt9rE2c/v2/5bYP8IwmeDvHFc9fGU8Oved/I6sLltXFNcqsu5xf7BP7I174m1iy+N3xI0c2uj2Eiy6HZzREG9nB+S4cHrGpOUJHzNg42rg/pZHbqCSepwR7D0qO3sra2jjht4khjiAVEjGAqjoMe1W/lHOK+WxOJliKjlI+3wWDhgqSpwEHHSiiisTsCiiigAzXzf4c/5Pt8X/8AYgWH/pUa+kO4r5v8Of8AJ93i8f8AUgWH/pSauHU5MTvD1Po1e1LtPrSL2p9QdYVwXx3/AOSKePf+xa1L/wBJpK72uC+O/wDyRTx9/wBizqX/AKTSU1qznr60pejPwrVwY057UzzB61Csv7tPpTd5BwOa6lE/N7XZ6d8Pf2d/jJ8V9EbxF8O/At7rmnpcNavPBJGoWVApdCHYdA68+9dP/wAMU/tS/wDRGtX/AO/9v/8AHK+2v+CXwV/2ftQU5GzxPdjg9R5MHFfZHlp/dFZSnyuyPqMLkdCvRjUcnd+Z+Lv/AAxV+1KOvwZ1c/8Abxbf/Haav7E37U56/BrVv/Ai2/8AjtftNtQfwCk2IP4RU+1Z0f6vYfu/w/yPxdj/AGGf2rZG/wCSRXyf795ar/7Vru/Bn/BNr9ovX5AviOPw/wCGoAcO91qAuJAPUJCGz+LCv1n8sZzgflRtjp+0Zccgwy3bfzPkL4Pf8E2/hB4Cni1fx/d3PjfUkO9YrqPybFD6eQrEv/wNmX2r60stLsNNtYbHTrSO1tbdQkUEChI41HRVUYAA9OlWgFH3Vxn0OKMD0rNyctz1MPhKWGjy0o2Ae/P1ooopHSMkPGf89K/ny8QN/wAT7VOet7L/AOhmv6DpBwR/npX89niA/wDE+1Tn/l8l/wDQzXp5Zq5HgZ7FSUPn+hXz71+oP/BKvDfBXxPz08Uv/wCktvX5b+bX6i/8EpDn4KeKD/1NUg/8lbeurMI2onm5NHlxXyPt/vmiiivBsfYkL2seCMnBP/fP0/z1r8of2/v2XT8JvGcnxT8Iaft8I+JbnNxbomFsL8nLJxwscuCU7ZDrx8gP6yHkYrl/iD8PvDXxK8H6t4I8V2S3mlaxbyW88efmXONrof4XQgMpHQjiunDVnQqKRwY/BRxtLle62PwGDvzlSNvrj6/j+FW7DUrrTLi3vbC4e1urWUT200R2vFICCGVhyCCAQe2K7L49fBnxF8BviPqnw+8Rh5TaMJrC78shb2zbIjmX0yOCOzqy9snzwOGGCa+ohy1IJrZnxEqUqU+XZo/W34CftOWP7QP7PHihNRlij8X6DoV5ba5aAhWlAtmCXSJ/ccAZ7K+5em3P5MxzBo0Yf54re+HnxD8S/DTX38ReFbsQXE9lcadcxfMY7iCaIxyJIoPI2kEejKuMYNc2m1EVPSsMPhPYSk1szsxeIeJjFS3RMzbuhxX69/8ABNz/AJNV8Pnv/aep/wDpS1fj+r571+v/APwTYbf+yr4f4/5imqf+lLVy5rG1Nep2ZGrYh+n6o+phxRRlfeivAPrgpkkQYY9yetPoHrR5h3Z+Nf7ePwmk+F37QetXFnbeVpXi5f7esTgYDuT56exWUOcf3ZFr5yM5PJWv1b/4KYfCj/hNfgcfHenWxfUvA94uoEovL2Um1JwSPT5H9MIa/J4OHHmJyh6c19Tl1X21HzR8NmmE9hiHpo9j2/8AY7+K8fwc+PvhjxTe3Qh0y9mOi6qCfkFtcHbvPsj+W/0Q1+2AuFwQSOMYweuQSP14r+eNCX3RljtIAz9Olfsd8Bv2i9K1b9kK1+MfiC98ybwlo89trKl+ZLu0TaoJ7tIBEw/66iuHNcO7xmuuh6OSYnkjKlLZao+K/wDgpP8AFE+M/jlH4Isbwyaf4KtEtCqtlWv5wHmP1CeWvPIKn3r5K38Yz+lXfE/ifVPF2vap4m1qbz9Q1m+n1G7cn780rFnP5k1mbiWxnivUwlD2NNRR4uLqPEVpVGekfs//AAzn+NPxl8LfDWBnMGpXiPfuvWOziBknOex2Aj/eIr917LT7WwtIbOziWGC3jSKONOFVVGFAHoAAK/PL/gld8IBHa+Jvjdqdvh7p/wCwtIdx/ApD3Mg9ywiXPqjjvX6LDgcV4OaVeerydj6jJcN7Gh7TrIKKKK8yx7QVUvD+5k/3H/kat1WvV/cP/uP/ACq4/ETP4T+ffVGB1e8JP/LzKT+BqNXx3qLVG/4mt4c/8vMv/oVQrKPWvt4x90/OqsPesfo//wAEoWD6H8SM8f6bp3/ouavvpYQqthsZ7CvgH/gk5Ju0P4kgf8/mm/8Aouav0Cr5LMFbESPt8sVsLH0Pj/8Abn/ZLj+LuhT/ABJ8Aab/AMVppFvmeGJPm1i1TJER9Z0x8hPJHyntt/J12ZZJY5gYmibayuCG3A4I2nngnB6YyBjNf0PvAjklup56Z59a/Nn/AIKF/siNZvffH34ZaSTE7GXxJp0CcKw630YA+62f3gA/iEnQk16OWY9r9xVenQ8zNcsi37emtep8C+dlc17D+y541+FXhL4nWH/C6PCGna74X1Bvs0kt9GZF06Qn5Z2TO0oAQHBBAQ5H3TnxMk43DcEwCCRycjIIHfj9Kd5ny4Y5QEBhnoAcivenTVSHK+p4NH91Pm3sf0G+G9M8M6Votpb+FLPTLPTCitbpYRokJQgFWXYApzkEMOoI9c1t/ZxgEHH0r82P+Cd37Xn9mXFj8AfiRqf+jSEx+GdRuHwAev2J2PTnJiJ6HKf3cfpMk5faU6MCR+n+NfF4uhOhUcZ6+Z9vg8RCtTUoK3kTIhHQ06he1GV9a50rnUFFFFUAUUUUAIeor5w8N8/t2+Lz/wBSDYf+lJr6Qr5v8Nkf8N2+Lx/1INh/6VGqh1OTE6OHqfRo4p272ptFSdY+uC+O3PwW8fD/AKlnVP8A0mkrvTXA/HT/AJIz49x/0LOqf+kslOO5jWX7qS8mfgyr/u057f0FKsgzknmqwlxGvH+cU1p2Ck46V6igfnsoan6xf8EtpPN/Z91MhcY8U3n/AKJt6+ytp9a+L/8AglOwf9njVDn/AJmm9/8ARNvX2lXm1NJtH3mXrlw0E+wyil2n1pKg7AowPSiigAooooAKKKKAGSH5Sf8APSv55tfcnXtU5/5fZf8A0M1/QzJ90/57V/O/r0n/ABPtU2/8/wBL/wChmvVytXcjwc71UPn+hW82v1L/AOCT5DfBLxQf+pqk/wDSS3r8rt/vX6of8EmQP+FH+Kf+xrk/9JbeurMtKNziydf7T8j7iooorwD6sKaeRtPT0p1Jt96APmv9tb9meD9oP4cStotuieMPD4ku9EnwFEx2jzLVm7CTgA9nAPGWz+Ms8N3Y3M9nf20lrPbTNbzQy8PHKpIZGXqCMMMeor+iqS2V8qXI3dOOlfmR/wAFKP2WV0PUZv2h/A+mbrG+lEXie1hX/UTZwl4MdA3yq/YNtY/eJr2MsxXJL2U3oeFmuCU17WCPg1XJGCMjoB2p3m9qqqzYdiQAvQdSc/8A1sfnUm/3r6BanzUoWJhJ7Zr9hP8AgmoQf2U9B9tV1Qf+TLV+OqkJ1Oa/Yf8A4Jpc/spaDg/8xXVP/Slq8rN1aivU9bJVbEP0/wAj6rwPSiiivnD6wKKKKAMrxD4b0zxPoeo+HdWhE9hqtpLZ3UR6PFIm1h+RNfgl8UvAV/8ACz4i+JPh5qyk3Ph7U5rMv0EiAny3x2Vl2tn0Za/oCwK/L/8A4Kp/CIaL440D4y6TbkW3iGE6NqjKo4uoBuiZj/tRcf8AbEeterlNbkrezezPHzigqlJVOsT4TDkdDXc6H8XvFGgfCfxN8HbKdf7D8VahZ6hdAs29GgHCp6ByI93/AFyX1OeA3e9Lv4x9e/rX0s6cZq0u58zCTi7x7ExfJORxU1ja3urXtnpWm2zz3l/cJa28KDLPK7KqLj1ZmAHviqO70PtX1Z/wTo+EJ+JP7QFn4m1C3Mmk+BLf+1p+Mq12fktkJ9QxZ+f+eRrPEVFQpOZeHoe2qKCP1C+BHwwtPg/8K/DHw6sAuNE0+OCZ0GBNcN880nuS5dh3wxr0fd7VBHDsG3vnOff1qWvi5ScpczPuKcVCKiug7d7Ubvam0VJY+qt8w+zucfwP/I1Z3e1V75T9nfn+B/8A0E01uTLY/nn1Rx/al5/18y/+hVXD47UuqN/xNLv/AK+pf51V8yvuoaxPhJxvI/Sf/gkuwbR/iXjtd6Z/6BNX6EV+eX/BJA7tH+Jf/X3pn/oE1fobXyGY/wC8SPr8tX+zRCqV7p8F1FJFcxJNFKrI8cihg6lcFTnqp7joau0EA8EVxrSzO6ys10Px8/br/ZNl+BHilvHXgbT5W8C+IboskaqWGl3bYJt24yqMNxQ46ZQ8qN3ygCSSSuOzLn0H8q/oN8c+AfC/xE8Kan4L8XaVDqGj6vA1vdW8nIZG6FT1UhvmBByCMg5r8TP2mf2dvE37NvxIufCOp+ZeaNdM1zoeplQFu7cHoxxgSIDh17HBHDCvp8rx7rL2dR6o+YzLAqg/aQWjPKIp5VeNo5pI3jfejqxDRtkEOpHQggYI6Yr9af2C/wBrmL40+Gx8O/Ht6n/Cb6FAreZIQp1a0XAFyo7yIAA+OTw+MZA/JDnk56jB9q3fBnjPxD4A8TaZ4y8JanJp+saPci7tLqPqsgb+IfxIRkFTnIJByK7cbg44um19pbHFg8VLCVObo9z+hZbhPlwchuQfbj/GpcD0rwn9lX9pjw3+0l8PIPENgsdnrti6W+uaYGG60ucfeUHnynALKfTIPKmvduvevjZ05UpOEt0fY06kasFOOwUUUVJYUUUUAFfNfhwkft4+LVzyfAFhn/wKNfSnevmnw7/yfl4t/wCyf2H/AKVGqh1OTE6uHqj6THPSihaKk6x272rg/jmP+LM+PP8AsWdU/wDSWSu6rgvjvKqfBfx4SDn/AIRnU+M+trJTjuZVv4bXkfgX5nyKMf5xSZyMZqFJInRWEq89OvNOyMgGQc+1euj4SUHc/Wf/AIJTAD9nXU2AwD4qvRj/ALY29faQr4y/4JW7R+znqAXLEeKr3JHQHyYK+za8qr8bPtcF/u8PQKbtPrTqKg6xlFLtPrRtPrQAmT/doyf7tOwKMCgBuT/dop2BSbT60ANkTKkZr+dbXiU1/VQ3a/l/9DNf0WkjBB64r+dXxTEIvEusqzHH9ozoOO4c162U6ykl5HiZz8MH6/oZ2/3r9V/+CTKkfA7xQc9fFcn/AKS29flLwf48fhX6tf8ABJx1T4GeKCSP+RqkxyP+fW2rszNWoHHlMbYi/kfcNFGV96K+dPpwooooAKyPEPhrSPE+k3uha/YQ32n6jbva3VvOm5JYXUKyEHsRnj3z1ArXop9UxWTTTPw1/a0/Zs1b9mv4mz6BEs1x4b1LfeeHb5h/rLcnBhc/89YydreuUbA3YHiO/wB6/dr9pn4AeHv2h/hlf+CNV8u31FSb3R74jLWV6q4R/UqeVYZ5Vj6DH4beLfDGu+BfE+seD/FtmdO1bQrprS9gkzlJFJBx2Knghs8qykZr6fL8Z9Yp8r3R8rmGDdCpzRWj2M7d71+xP/BMwk/so6Fk/wDMW1T/ANKGr8chnPEiEfjX7Gf8EzCB+yloQDAn+1tU4/7eGrPN/wCCr9zTJ42rv0PrCiiivmz6cKKKKACvFP2uPhCvxo+Ani3wbawq+oJatqGlkjkXkH7yMD/f2tH9JDXtdRyxZ6H3+nGKuE3TmpLoRUpqpBxkfzl5YAbwQc7SvcMOvHoBijd717h+2p8JZPhD+0X4p0S2shBpWszDXNIJXapguCSyL7JIZE+iivCt43f6xP8AvoV9tSqKrBTj1PjKtJ05uNtiVHC4yf4d2OnAHXPTFfsh/wAE7fg+3wz/AGftO1rUrQw6x44f+3LvcuHEBULbIf8AtmVfHrI1flR8CfhldfGD4xeFPhnbq7Q6zfxLdOp3GOzQb524xgBFf8R7iv320+wtdOs4bGygSKC3jSKKNBhURQAqgegAA/CvGznEPljSXU9fKMNZuq0WuPSkopdp9a+fPfEopdp9aNp9aVwDafWob4D7NIfRH/8AQTViq9+QLOY9gj/+gmqW5MldH87erHGqXmD/AMvMv86qc/3qs6sU/tO9Hmci8lTG1jzn6VS3g/xevYivuqTTifES3ufpT/wSKbOjfEzI/wCXzSx/5Dmr9D8n+7X54/8ABIsbdF+JZw5zfaYAdnH+rm61+h+R6V8jmP8AvEj6zAf7vESiiiuI7Q9smvJ/2iPgJ4V/aG+Hd74E8RxiGcHz9L1BEDS6ddqvySLjHHZl6MpIznbj1imGNWG1jkN1B6VVOTpy51utiKlONWPJLZn8+/xN+Gni/wCEnjbVPAPjPTWtNT0uTa+OY5kJPlzRt/FG64ZT6ZBwQccoQW/iI+n+fev2R/bg/ZVtf2ifAp1Lw/BFH438OI0mlz4C/aYx8zWcjdlcjKk52vjoGNfjrqNhf6XeXemapZS2V7YTNb3NtcKY5IpVO0oyn5gQ3GMfzBr7HAY6OKppv4kfJYzAyw1Rrod78B/jr4q/Z++JNl8QvCx87yysGo6ezYj1C0ON8TejYClX7MqnpxX7g/Cj4r+EPjH4G0nx94J1BbzTNXi3oQRvhccPFIP4XQ5DA9D65BP8+6hhjbgY6fKeK+lv2Jv2pL79nHxwtlr8kk/gXxDOg1S2V8tZyjIW8jGeqg4df4lz1KiuXNMAsRH21P4lv5nTluLeHl7Oo9GftLu9qWs/TtWstXsrfU9LuobqzvI0mt54pAySxsMhlI4YYIxitCvltnY+nTurhRRRQAd6+a/Dqf8AGePi0/8AVP7D/wBKjX0pXzZ4e4/bv8Wkf9CBp4/8mmqodTlxG8PVH0gval2n1pF7U+pOobtPrWP4p8MaZ4x8Pal4Y1oO9jq1nPYXKI21jHKhRwGHIO0kZraowPzoWjE1e6Z8ix/8EuP2VY8bdM8UnHQf25J/hUrf8EwP2Vz/AMw7xOP+43J/hX1tx6UmT1q/aT7mLw1J/ZR5t8DvgJ4E/Z58IzeCvh7/AGiumT38uosL2585/OkVVPzYHGEXj2r0mk2j0pah6u7NYxUVZBRRRQUFFFFABRRRQAUUUUADDdXyTef8Ew/2ZL+8uL65i8VGW4leZ9usEDczZz92vraggHrVwqTp/A7Gc6UKnxq58in/AIJdfsvH/ll4t/8AByf/AIivavgT+z54C/Z28NXfhP4evqg0+/vzqMi3915zecY1Q4OB8uIwcV6hTMU51qk1aTuKFCnB3irMMpRTtq+lNrM1Cil2n1o2n1oASiiigBCoOcjr196+e/jh+w18Cfj94uXx540stWg1n7PHazXOm35g86NCSvmLghmCnbnGcADoBX0LS8cEjpVwqSpu8HYmcI1FyyPj1P8Agld+y+ibfM8Yk/8AYY/+119C/Bf4M+EPgP4Ftfh54Gk1A6RaXE9zGLyfzZA0rFnG7AGMknpXefhRgegpzrVKitN3IhRp03eKsGV96KMD0orM1Ciil2n1oASlyD2pKKAPFvj7+yV8Jf2kZtJvPiNBqYudFEyW0+n3QgfZIQWR8hsgEAj05x1NeQr/AMEqv2Ygcm68Z/8Ag5X/AONV9j4/Sl4xjFbRxFSCsnYxlh6c3do+fPgd+w98Ev2fvF83jnwMuuXGqy2jWSPqd8JxFGzBn2AIuCdqjPpx3NfQYKgYBpMCjB9aznKVR80ndmkYKmrQF2n1p1FFSUFFFFTYAqOaFJkaFzwwIP0NSUVQHyDP/wAEuv2XJ7uS8kt/FQklkMjAawwGT1PC1J/w69/Zd72/i0/XW3/+Ir647YpK3WKrL7T+8weFo/ynknwG/Zj+GP7Odvq9r8OE1RY9ceGW7+3XhmJaIMEI4GOHP5V65tPrQFI5Bp1ZSk5vmkzWMVBWWw3afWjafWnUVGpQ3afWjafWnUUwIWhDggsR1r56+Ln7CP7PPxp8W3HjjxboWo2+rXaqtzNpuoNbCdl4DuoyC+04LYyQozmvonafWkqqdSdNtxdiJU4zVpq58jxf8Euv2Voeth4qP11t/wDCpk/4Ji/ssoCFsPE4yMf8hpunGB93p8o4r6z2n1pdq+lbfW6/8z+8y+qUbWcV9xxHwn+Enhr4N+DrTwN4TutUm0mwZjaJqF0bhoEY7vLVyAQm45A7dBgV3FGBnOOaKwbbepukoqyCiiikMK+bPDvP7eHi0H/oQdP/APSpq+k6+bPDpH/DeXi9f+pA08/+TTVcOpy4jeHqj6QXtT6Yvan1B1BRkZ545xRXnnx4+J9z8HPhP4k+JNrpKanLoNqLlbR5TGspLomCwBK/f9O1C1diJTUIuT6HoW8en6UZxn/Gvz607/gpZ8X9b0xdY0H9mS41OzYsontLi7mQsvVdyW5XPrzxXcfAf/go14a+JXj+1+GPxH8D3vgXX7+TyLUXMheCSc8rExdFeJyMYBUgkgZGRVunJI5aeYUKjtFn2ceOtITjtVO71K2s7Rr27uI4YYomuJZJWCKkajLMT2AHX0GfSvkT4F/8FCtE+Mvxzn+E48Kppmn3s91HomqNdl2uzHlo1aMoNpdFYjBOCQDmpSbvY2q4iFJqMnvoj7IpM8cDNND8Alhk9q+cPD37XV1rn7Wmr/szN4Iigt9LSVv7Y+2sWfZbpMf3Xl4/ix9+hK5U6sabSl1PpLIHrSZ46ivK/j/+0J4K/Z48Bz+MvF85aRiYdP05GCz38+MiNM8Ad2c8KOvOAcL9mv40/ET4z/D5/iH43+Hdt4Ttb1jJo8QvXmkvLcDPnFSilUP8J53DJwOMlnuJ14Kfs+trnuG70BoMijsT9K+df2SP2o7/APaVj8Wy33g2Dw//AMI1cQQII717jzvMEmScxptx5f8A49XGfGr9vlfBvj27+GHwh+Htz4613T5DBdyRu4hSdfvxRrErPKUyN54CnjJNPlexnLG0YQVST0Z9fbh3BH1oJz90E18I+Bf+ClOsQ+LLbwl8b/gprHheW8lWKKW0jmklBY4BNvKiuy5wMoWPPQnivWP2l/2tfFfwE8S6PoOg/By/8Vwapp329ryGSWJYj5jJ5ZCwvzwDyQfmHFPkd7ErHUZQ509F5H0tk/3T+dIXHG0Zz0r87j/wVP8AGq6wPDzfs4XC6owUiyOozfaCGBYERfZt54GenTnpXr37Of7ZHxA+NfxJHgfxD8C9Q8J2bafNff2ncyTsuYyPk2tCg53dd3bpQ6bSuEMfQm0k9z6z3jAJ4J7ZpCR2714l+1X+0dZfs2fDePxemkw6tqt/eR2Wm6fLK0Qnc5Z2LBTwqDPuSBxkGk/ZR/aPtv2lfh1N4wl0iLSNT06/ksb7T4pzKIzgNG4JUHa6sMe4Poamztc1+sU/a+xvrue378dQfzoIwa5j4j+Ln8BfD/xH43jsBeNoGlXOpfZy+3zfKiZ9u4A7c7euDXhP7Kf7bOh/tI3WpeHNT0KPw54lslFzBYfafNW6tehkRiqklTjcuM4IbpnByvcc68Kc1Tk9XsfTxcDAoz/nNcD8aPiPN8K/hR4m+I1vpS38vh/TZL5bZ5DGJWXHylsZXr6Vy37Kvx31L9of4Vp8RdS8NQaHK+oXNl9kiuDOMRFcPuKg87vSi11cfto86p9T2bafWgc4J6U4c18g2X/BQDSLT9pXU/gT438KQ6LYWuqzaNba39tLKZ1YiMyoUARXIA+9xuB6Zw4xctgq14Ubc7tc+vMD+9SbgeBk4qH7QoTcWGcjK5xjPr9MjNfNXwP/AGutR+Lvx38f/ByfwTBpkHgz7T5d8t68r3Xl3f2cZj8tQufvcMfT3pJXCdaNNpPqfTeVx1HNGR618M/HH/goP4v+Ffxs1n4ReHvg0niWTSvI8qWG+l82cPbrKxESRN03HoW+6axbT/gpB8ZSMyfsnayT6eddj/21qvZy7HPLH0IOzf4M/QHA9TScZ6ivDtC/aF8Q3/7NOpfHfXfh7JompWGn31//AGDdTOshWB2VMu8asA4UHOz+IcVo/sv/AB4u/wBob4Xp8Q73w3Focrajc2BtI7g3A/dbfm3FFPO77uPxqbG6rwclHq1c9h46k4/CnZGccV5d+0Z8ZpPgN8INd+KUeirq76MICLNrjyRJ5kyRH5trbcb89D0rnvhV+0ZdfEv9m+T48nwvHp8w03U9QXTBcmRc2hlGzzCi9fK+9twM9DQldXCVeEZcr3tc9wIGetJg96/PLwv/AMFPPih4nt/tWjfs2T6hCG2PJZ3dzOsbYBwWS3Izg9PcV2fgn/gplor+JbXw78Yvhdq3ghbtgovXd5I4cnAeWN40cJxyy7sD25qvZyOeOY4eTtzH23gf3qaT0wCc+2aoSa3p0GnvqtxfQx2cVubqS4LgRJCBkuXPG0AE59Bmvib4if8ABS/Sh4nm8K/BT4a3fjMxllF7I7Rx3BUnJghRWldOOHIUHBwKlJs2rYqlQXvs+6cn+63XHSivg3wJ/wAFMHsvEtt4d+Ovwkv/AAhHdEMLyDzXEIJ+88MiK+wd2Utx0Bwa+3tG8R6Vr+mWetaNqFve2OoxCe0ngkDxzIRkMrDtjn/OA5RcRUMXRxF+R7Gtk9dv5mk3j359u3rXzB+0j+3Z8P8A4Cau/gqy0yXxN4tVBJJYwTCOG1DDKiaXDbZCOkaqzdM7QQa8XT/gox8ddAjXXfGP7NlxZ6AzKxneO9tjsPfzJEZB9SMH2pqnJq5lUzHD0pOMnt5H6E0ZHWvLfgT+0P4H/aD8KHxP4KF9C1uyxXtpeWzRtbylclPMx5b8YOVJ4IzgnA+aPiF/wUc8WeEfi14o+F2gfAifxBP4c1CezD22oSmSVI22+aY0gfaCccZxyOaSi27Gk8ZRhBVL6M+6Mge9JvHpz6Z5r4FH/BSH4uj/AJtN1s8drm6/+Ra+gfG37R2ueD/2Y4fj9P4BKai1nZ3c2hXNy0bwmeVIzGzlNwKl/wC72punJdBQx1Cabi9vJnvHvzj60uD3XH414N+yv+1V4f8A2lvDF5dR6Wui+IdJkxqOked5rRxMf3cqMQCyNgrnHDKQeqk7H7Tvx4uP2evhU/xIs/D0euMl/b2n2WScwgrKW+bcFJ42+lTZ3saLEU5U/ap+6exFgKTcD057cVxXwi+IEvxR+GHhj4hT6aunP4i06K+NqknmCEuuSu4gbgPXA6184fG//go34F+HviaTwN8NPCtz491m3la3nltrgR2qzjgojBXaUjn7i44OCaag27BPE06cFUk9Ht5n2Nx17f1pSMGvgTRP+Cnmu6HqsFl8ZfgFrPhu1uGUie3eQyKhxlhHMibwM8kMPoa+z/AHxJ8K/Ezwvp3jLwbq0Op6TqaFoLiLOMg4KsMZVgwKspGVI59aHBomji6Nd2g9Tq+BzmivifWP+Ch//CHftPaj8EvGvhGxsPDtlq40o64Ltt0W5cLJIpG3aGK7sHgetfaC3QI3A5BI2kd84x/X8qTTRdOvCq2ovbQn3YySKaeK+Q/2rP29U/Z78dW/gHw34StPEuow2i3Wq+ZdtCtqXIKR8K3zFCGOegZfWvZPi18bLz4Y/AST4zW/h5NQuEs7C5GntOUUm4eJSN4UnjzPTtT5Xa5P1qleSv8ADueslwO1G/oMdT/WvOfgN8VLr4yfCXQPiXe6Kmlya1HLIbSOUyrEEmePG8gEk7M/d7gV4L+0H/wUQ8D/AAn8TT+APAPhyXxr4itpWhuPImEdrBMG5i3gM0kgOQVVeCCN2QaFFt2HPEU4QVRvRn2Bv5xtbn/Zp2QPevz10v8A4KX+P/C97BL8Yf2d9T0TSp5BtuIHmjdVPcLPGFf0xvHPrX2n8LfjB4J+MnhK08a+ANWGo6XdMYg2wq8Uo+9FIrYKuvcf0INDi4k0cVSrvlg9Tt6KKKk6Qr5p8O/8n8+Lz/1T7T//AEqavpavmrw2P+M+PFy/9U9sP/So1cOpy4jRw9T6TVTxzTqB2oqDqCvBf25Ilb9lP4icDKaYmM8gfv4v8K95rwT9ul2/4ZR+IqqcbtLQ9OP9fDTj8SRz4lL2Mr9meb/8E7df0HSP2XNKhvda022mbVdRZhPdIhXMp5Kk8/TjNfOH/BRDxB4L8ffHf4caX8LtRstc8XWgS0u5tLlWZhIZ4zaxF0J3OG80hdxKhs9xVX9lD9gD4fftBfBjT/iR4h8Z+IdNury8urZobWOCSPEchRTl1LdOTk19j/A39hT4HfAjW7fxZo9jqeta9briDUdVmWVrYEc+VGqqqMSSA20sMnB61tzKLueZSpVa9GEErR01MH9vf4wzfDb9nqbRLO8aPWvGrf2LEEIDrbbS11IOehj+Qn/psO9fLfxo/Zy1b9nX4IfBv44+GoGtPE3h64juNbcLhluZ5PtMG8nHCMDAR33KKd+0lqnxL/a2/a2m8KfBrTbPWYvhxGY7OO6dPsjNbyr9qkcyYQhrhkj25IdY19a7v4i+A/8Agpn8TfBOoeBPGel+FdS0bV4wlzF5unR8AhgVdSCpDgEEYIxwaqC5UtTDES9vOUrN2+G3R9z7t+HvjnRviR4G8P8AjzRJ0ay12xivYArbiu8AmMkd1YlT7g+hr87L/wCIXhb4Tf8ABRzx94/8X6k9tpWkwXUkpVd0kjNYRKkcajBaRnZUVc8k89Ca9E/4Ji/Fa9k0XxJ+z74sEkWreDr2S6sbeYkMkDSlbmLGekc+cj/pt6CvJvEfwj8O/G//AIKSeJfBviSO4XSWvGvr2NN2+4SGziYwg9g5CqSMELuwQSKzS5G0dFes61GlL7V195rfA20b9v8A/aO1b4ifFLUIV8NeDfLax8KCYMzxF2MaFR1j3KTK4zvchO/H6VXVnbWmkywW8QiSO2fYigBQoQgKAOAAAAB6DFfm5+0z8M/Ef7F/xq0P9pj4K6YsHhS7uBaalpUSmO1iLAB7faPuwzBcr/ckU4x8mfvjwD8UvCfxe+GVp8Q/CF01xpuq2UkoDAb7dwp3xSDs6NlSPb6EzLXVGmEcYOcJ/H1813R8b/8ABKItJb/FJzvG2/0/pjuk5PSvM/B3xK1D9hj9o/xyfib4BvNS0/xDPcAX8ahZTbtcNJHPbswCyK+751yDuHJBUg+n/wDBKdbeHS/ijcECGP7bpzElsYAimyTnoMV79ZftK/sofGfU9X+Hup+JvC2qzWN6bJrPXFiEV2y8ebbtMPLlQnIVlOeM8Ag1TbUnoYUaSqUKclKzV7E3w6/aQ/Zb+P8AqWjQ6Trmj3eu6ZOt1pdlq9oIby2uNh5gEgx5gAOfLJ7dTXugTaCCuNuSA3pgce/19q/LP/goD8L/ANn74WW2h+J/g1eW+i+Lb283nTNKuyY/JCkrOqAloijBQpUhTv8Au9a/Sf4ZXmval4A8NX3iaIx6tcaTaG+DZDC4MKGTtxht1RLa8TuwtaUpShUSuuq6nwT423Sf8FYvDsWMoFtcjgf8wyT/AAFfo2EVYQPu5QDA7E/Svz08T2Jf/gqzolyUVlRLY5PPTSpDX2Z8efijZfBz4SeJviHMyGXSrN1tI2OPNu3+SFOeuZCv/AcnsacrtRROEnCPtZP7Lf4Hxt8VpB+1d+3doPwogK3HhH4bktqAxujZ4ir3WRyPmkEVv74btU/7P1wn7LX7bPiX4JahILfQfHJP9llgI0Zm3z2hUZ54aWAercV5D+zD4I/bU0zTdQ+LnwJ0TS5U8XzNFc3+ova+dP5Ur7yFmbO1pC+SAMkDsvON+1D4a/bcibTPjh8bdBsbVvCUkMcOq6W1t5kDeeGhLiBslRLgqSCATgnBxW9rPlvoeQpSb+scr5ua/lbsfpp+0hJ5P7P/AMRn4OzwrqeSTxk274P1r8ovhv8AD74ieGfhBp/7WXw4vrn7b4U8RzWt8iDeIY0WIpOVH3oiZnikXH3WTtkj9G/E/wAUrD40fsReJfiTpsa7Nc8EXs08IOfKuRA6yxcf3ZFdfoue4rz7/gmhp9hqX7L+radrOnwS2l74jvre4glAdZY2hgRwwIwwIJUjpg1nF8iud2KpxxVeKT3jv2Nfx78b/Dfx/wD2GfHfxB8PkQyyaBNbajZkgvZXQC74mx2HDKepVlOATtF3/gmpx+zPB8x51zUf/QlFfH/7RPgTxh+xT4j8ZeFvDCT3/wANPippc1lao7HEJxuRCxyPNgY/KTy8UnPJO36//wCCaO4fsw2BdcGTW9S6+vmjj9KbSjC6JwtSU8Uoz+KKsz6wLBO1fjl47+EWq/HP9rT4y+D/AA7ctHrdhcapq2mQ7wEmmhnj/df7JdXZQexK9jX7FyFgM4zxX5qfs4qt1/wUs+K0flYxHrJLjAPNzACOc1FKXLc6sfBVJQhLqz1T9gv9qy5+JWhyfBr4i3ckPjvwsohQ3R2SX1pG20lg3zedHjZIOpwG7tjzz9h+NLz9tL44zlTtcagdvGedTx/LitT9uX9nPWfAHiOz/a8+CYfTta0S4jvNeigXJyOPtu0fewMJMP4kYEg/MTw//BNHxJJ4w/aT+Jvi+8sjayazpT6g0MTFxE8t8r7V7kZbjI6Vas4uSOO8oYiFOp9m9iH4o+P/AAn8K/8AgpHf+O/FU94un6X9naYW9sZJsNpaxqAvBY7mXPpknnrX0ef+Ck/7MUeWe/8AFR2gsR/Y7/y3V4d4k03wdq//AAU11C0+IFvot5oDxq1zHqhja0YDSRsLCT5OG2kZ9B3r63Pw9/YvIMbeGfhJtbqNlhipaj1QUlW56jpyS1e4v7SWrQat+yp448QaYS1tf+E572ESDYxWSMMCR2O1hXnH/BM2YzfszRMcYOv6jnuAPlr1j40aNY/Ej9nzxl4W+H82n6hFfeH7zTtPjsp0kh81YmCRKUJUHcFT2PWvln/gmR8a/BWkfDfUPg54l1e20XX9P1ea6toL2ZYBcxShchSxwXUq25PvAYOOoqbXjodMpRp4qEpPpY9t/wCChjE/sl+Nt4PAssAdf+PyGuJ/ZYX/AI12TOUxnw54hbn/AH7usn/gpL8dvAFp8EL34Zab4msNR8QeIrm2UWlpcxyvb28cqyNLJtJ2j92FGcElsjIBI7z4F+C9T8EfsFReGNbsprfUJfBup3MsMo2tGbmKeZUIPQhZAD6EU0rRs+5U2qleUl0iePf8EzfHfgzwj8IvEtl4m8XaFpc0+vF4oL3UY4HYfZoQGCuwJGe4Has//gpR8WPg94o8I+GvCfh7XNJ8QeJoL/7VHPp00dx9ltREwYO6HC722YTPJj6cLngP2Nf2OPhP+0R8Pte8S+OZ9ft77S9aWwtzY3kaqEEET8ho2wdznn0IG2qPw88A+CP2R/2wYfB/xp8LW+qeHL6VZvDet6kDss974gumXPlnDfu5GKnYfnUKAKu9p3TPOUnPDRg17r6nsf7U3jHxh8M/2Gvhz4C1k3FrrPiOwstM1XcSHS3htwzwtycMT5cbf8DHQ4r6G/ZI+Bfhb4O/CLw9/Z+j2663rVjFqGragVUyTTTIHKFsZ8tQQir0+XJ5JJ4f/goZ8J9Z+JPwEGreG7SW4vvCF9/afkRruaazKFJsAcuVDI7evlNirX7Gv7Wvw6+Jnwt8P+FNd8UWGleLNCsYdNvLK+nWFrkRJsW4jLY3hlUE45DFgcYzUNtxujsp8tPF2qP7Kt5nqH7RPwX8L/Gb4U6z4d1/TreTULeylutLvTGPMsrtFLJIjDkDIAYDqCa+Wv8Agnt8X9QtfgX8RNEu5pbyP4fxyarYhySFie3kmaEYzhd8TnHq7V7p+1P+1H8PfhF8Ndcs7DxPp2oeKdTs5bXS9PtLlZZEkkUgTSKp+SNOWJOM4wOua8p/4J+/A3UdF+AnirV/EdncQSfEfMUUbH5xp4iaJHIOOW3zuB/d2+oprm5LSHWUZ42LpbpO/wClzzL/AIJsfDDTfih4o8Z/tFePoDrGvWmpiGxluVEuy5mUyzXAz/GfMUKf4QWxjIr7Y+IHx8+Bfw41VvCHxM8f6Lo9/c2y3Bsr0HdJAzMobbtwykqwwfQ18L/sR/FvS/2V/ij44/Z5+NE66ALu9RrXULhilubmIbAS/RY5Y/LZJD8uAAT82R9Z/GH4XfspfEXUI/ib8XP+EevGtLJYU1CbXWhi+zqzMoOyVVAy7HI5570pXUvIdB2o+4lzdbnUfA74lfs7+INPk8BfATxFodxaaMpu3sdJDhIVklJ3MSuPmkLcEn8hXxB4N+MPgT4J/t5/FDxl49vLxLBrrVbQPa2bTuJHmjKjavJG2N+enAq5/wAEv9Q0U/Hn4q2/h94f7Ia1A09wzAtALp/KILDn5O5x169q1Pg9pXgPxH+318W7D4j6R4bv9LWbVXRNXSGSLzRcQhWUScB8E9OfvVaVpM5q1SVWFKWifNp2PoDS/wDgof8Asx6xqVrpVlrHiGS4vJoreFTosw3NI4Re3Tcw5rX/AG9bYr+yt42ZWLfNYbdwzhftkPFdVH8L/wBlC2njuLfwb8MopYnWaN1gswVdTkMOOCDzXFft8XkN/wDsjeMr7TrqOeGY6e0ckMuUb/TYOhXI9s5IqLR5lZHbNTdCp7Rp6dD4Q8J+HviF+zH4f+Gn7WXg2W51DSNdjltdZtnY7DIs8sTW8pHGyWKNSjH7roec4FfT37dPxH8NfFD9jDTvHHgu9jvNN1nWNOkgzw0bAuDG47OrAqy9iD1HJ7n9lPwV4c+J37EPhTwT4v0+O60nWbG9tLiJWJO03k21lYA4ZThw3ZwpGK/Pf45eHfiX+zdD4l/Za8RF9Q8MalrNrr+lXbhwkix7lSaIcgeYpCSL/C8WB1y135mkec4PDUNPgkvuZ9peIfiXrXws/wCCaug67o8rW+pXfhjT9Mt5YgymI3BWN3B9fLZyPcVrf8E5vgl4X0P4O2HxYvdLgufEfil5JI7yRAz29mkhRI4yeUB2bjjGSwByAAKdz8Jr34zf8E5/DXgrw/ayy6nJ4asbyxhxzJcQESJHnPG/YUGeAXGeK5z/AIJ7ftPeCNF+H0fwO+Ius23h7XvDtzLDYPqLmCO5hd93l7nwFlRnZSjYJGwjJBAhqXLpudUeRV6cqu3LpfY+y/iV8M/B/wAVfB+oeB/GukxX1jqUbKGaMb4HA+WSM/wOvBUjBr4N/wCCcPivWfCnxQ+J37PGpXRnt9Lma+gDLlI7i2uPs0zKOwcGLOOfk+tfYHxm/af+EXwZ8K3viLWPGGlXF1HC4stOtbuOWe8mCnaiqrEjkAFiMKASTivk/wD4JlfDfxDrfirx5+0T4ms5ETxA0ljaOY/lnlefz7h1HUqHEa5HfcAeM04KXLZ7GtdwliYOnv1seFfFL4Tav8bv2q/jZ4b8OxvNrOltd6vYwAE/aZIDDiLnqWRnVemGI9q+lP2af21LCD9lvVNZ8fXxk8U/DeGOxNrM22XU1cFbM8/MWY/u2PVfLZz1rL/Z40+L/h4/8XJzaEjytUbKdT++taqfHf8AYL1vxF+0zpuoeB9Lmg8E+M7htQ1qe3YIumurA3J29PnB3xcffkZRgLVJp+5I44KpTvWoK7u1/wAE+V/iv8P/ABre/DrS/wBobx3eSPf/ABH1q/kiWQYMkKoG87/dZyyoOmxARwwr9G/2tFSP9h66VmwF0zQ1znt51uK8i/4Kk6RpuifCn4b6Toemxw2Gm3txaW0KKVWKJLQKiKCOgCqPwFem/tm3hT9hi9MSMW/szRDjGf8Al4tv8abaaiTTpexdaG/u/foc14K+I158L/8Agmtb+MtGuGi1G10S6itZ42wYpp794FfI6EGTf9Vqh/wTS+CvhmD4ZzfGnVNNivPEPiC/nt7W6nQO1pawuU2Rk8qXdHLsOSCozgYqX4M/Di5+Mf8AwTq0/wCHloxt7rVdGvIrfIwv2lbyWSAsew8yNAfbNcF/wT2/aW8L/C3SNV/Z4+MV5/wi+saXq0p02fVCYYgXP721lkbAjdJdxG7AO8gHNZtWvyvU6YKMZUvabcvyufoF4r8EeFfHfh+98LeL9FtdW0rUUMdxa3UQeN1Ix0PQjsRgg8gg1W8BfDfwh8MPDVl4R8D6RHpekaemyG2iJ2n1ZiSS7HuzEk9zXMfEf9o/4N/C7w3P4l8S+PdIaNIy0NrZ3cc9xcsASEijUksTjGcBR3Iq18EPjd4R+PHgW28eeDo75LSad7aWG6h2SQzp99CRlWxx8ykjnrnNQ4ytc9JTpc/Ldcx6JRRRSNwr5p8N/wDJ/Xi8/wDVPtPH/k01fS1fNXhsZ/b48YH/AKp9p/8A6VtVQ6nLid4ep9KjtRSK3y5xS1J1BXF/F34XaJ8Y/h9rPw68RXd5badrcIhnktHVZAodXBUsCOqDqK7SkI709tSZRU1ZnnXwQ+CXhj4C+ALL4d+E76/utNspridZL5laVmlfe25lVR+ldvf6e15Y3VnFdzWslzFIizxY8yIsuA6543KeRmr2BnNG0DPHXrRcIwUI8q2PEv2ff2Uvh5+zi+u3Pg6/1bULrxAYRc3WpyRyTIke/aisiLgFpGY5zk4PavaDbRtnJY565PWpNpA4AGOlLk45pN9yYU4042ijwLRv2Mvhn4d+P93+0VoGsa9p+v3881xc2cFxGLKUzR7JAU2bsMTvPzfe5q9of7KXgnQP2htR/aNtPEGuNr2ppKsti8kRsxvhSI4XZv4CA/f617fgdefzo2jOcU+dk/V6emm2vzOb8ceAfDnxC8Jat4J8VWS32l6zA8FxE4yRuGAyn+FlIDA9iAR0FeafAL9lTwn+zvpWtaF4S8XeJL/TdcxJLaalPG8cUoXa0kYVFIZl2g887FznFe37R6UHA5FF7aIqVKEpc7Wvc8Z/Z/8A2WfAH7Odpr1j4Q1HVr+HxG8Ml2upyxy8orKoUqi8EORg+leb/GX/AIJtfs7fFrVJNfs9Ov8AwhqspLSzaEyRwyE9S0DqY8+6Ba+rtp7jH0pe2KFJ3epLw9NxULaHx/8ACb/gmR8CPhrr1n4k1jUdZ8W3dhIk1tDqTolqsiHKsYowN+D2Yke1fXSWwjG1WO0HIBOf51NR160nJvcqlShR+BWueKXv7K/ge9/aBtv2jJNc1lfEFqFVbVZY/suFtzAPl2bvuknr1q/+0J+zr4e/aL8KWXg7xV4n1zTLCyvvtxGmyIhnk2Mq7w6sCBvY49/pXreB0oAA4HGKHLawvYQs4paP8TmfA3gHQ/h74N0bwL4aRodK0OySxtkIBO1VxuYjqxJ3Zx1Oag+Inw38O/E3wLrXw/8AEyO2ma/ZzWlz5e0ON4xvUkH5lO1lOOCoJrrQoBzjn1oIBOe9F9SlTjy8ttDwr4Z/skeCvhb8H/EPwT0XxR4iutA8RpcpL9snjaS38+HypDEVQBeBuwQRu5rpvgH8BPDH7Pngc+BPCOr6lfWLX8uo+ZfOjy+ZIqKRlFA2/ID0716eAp5ow2MYpt3RMaEItSS1PPPjR8EfB3x08BX3w+8bxyvZ3TLNDPFgTWsyklJIiRwy89eMMR0pvwL+CXhn4B/D62+HXhPUNQvdPtrie5WW+dXlLytublVAxzjpXooGOhNGB6UtUrDVOHPzpa9wOTx2rwjwN+yJ4F8AfHPxJ8fNI1vWH1rxMLkXNrK0Zto/PkSR9mFDcFBjJ717xj3pCcjpmhPcJQjNrmWqKV/o9lqlrNYalAl1aXERinglUNHKpGCGU8EEcEehIrxX4Gfse/DL9nzxv4j8b+AbzVxN4iiNu9lcTI8FtF5okVYgF3ADAX5ieBXvHvRgYxSi2tBypxnJSkrtHy98W/8Agn38HvjR8QdS+JHinX/FEWpaqsazR211EsShI0jGA0ZbkIDyepNcVJ/wSg/Z8c5/4SbxiP8At7g/+NV9q4GM44NHAqudoweDoyd3Hc82+BXwM8M/s/8AgCD4deENQ1G702C7nuxJfsjylpWyy5RVGOBzjPevJ/jV/wAE9fgn8YfEFz4vifUvC2tXrF7qXSfLFvcuRy7wupXee7KVyck8mvqHApcD0oUmnoaTw9OcOSS0PkP4V/8ABND4E/DrxBa+J9avdX8XXdi6y28WptGtqkq5w/koAHIz0csvtX0/4j8LWHifw5qXhi9kljtdUsp7GVom2usciFHIznn5j7VuAAc45/Wl4PBXj60ObYU6EKaaijyT4B/s5+Df2d/C2o+FPBWpardWmqXx1CaTUXSSQOYkjwGVV4xGDzzms39ov9k74c/tM6Tpen+OJtQs7nRbhp7W+sWQThWXDxnerKyMQpxjOVWvaghHOcGnbVJ5XvmlzN6g6FPl9nb3Uct4B8CweA/B2l+C49d1TWbbSbYWkNzqkiS3EkS8KsjKqh9o+XOMkda+dvi5/wAE2vgR8TNZuPEejvqfg7ULp2kmXSSn2V3b7zeQ6lVJx/CRX1kVC9uv60uOmO3SkpNaoJ4enUSjJaLY+R/hl/wTV+BHgPU4da8QXWq+MLi3KvFDqbItsGXoWiQDf16MSD3FfVkWn2sEAs7ePyYkULGicBFAwAo7AAAADoBVrOTtxz9aXgDAOKpyctApUIUP4atc8Y+Ov7I/wX/aGijn8e6DIuqQJ5dvq9g/k3kS9l3gEOo5wrAgZOAM186Qf8EkfhI2opPq/wATfFl5YRNlbRVt4mx6GTYfzC5r7ywPSjaOp601KSVkTUwtGo7yR518HvgF8LvgV4e/4Rz4aeGoNMhkO+4m/wBZPcv3eSRssx+pwO2K8J8f/wDBNX4OfETxzrvjzWPFfimG+8QX82oXEUE0AiV3YkhQ0ZbAz619d7T6UY7Y4oUnvcqeHpziotaI+J3/AOCUPwFkOW8XeMQf+vi3/wDjVe6337MPgLUP2fbX9m641LWF8NWlvBbJcRzIl2VjnEqksE253Afw17JgelG0Hg1HO3uSsLSimkjhvhJ8J9A+Dfw+0n4beGr69utL0dJEgkvZFeZ98jyEsyhQeXPQDoK5D9or9lf4cftLeHrPRPHEl/aXGnXH2i11KyZFuYwRh4wXVlKNxlSDyuetez4FBXuBVXaLdKE4qDWi6HIfDP4baR8Lvh/ofw70S8urjT9As0s4JbplaZkXj5yoAz+FePfHP9g/4F/HLU7jxLqGm3vh7xDdkNcalo8qxtcN/eliZSjn/awGPrX0jwabsP50uZp3QSo06keWS0R8X+Df+CWHwH8P6tFqnibxB4j8TJCdy2tzOsML/wC8IwGYc4xu6cdK+vND8M6H4Z0m00Lw/pkOn6dYRrDa2tugSKFAMBVUcKME9PU+ta2KCoPBWm5vqKlQp0V+7Vjx/wAG/sw/D3wR8Z/EXx10W/1k+IfEq3C3sc10j2q+ayMwRAoIwY1/ir1o2yGXzyTv2bOvFTYo2jufai99y4QjTVoo8f8A2hP2ZvAv7SeiaRoXjrUNYtrfRrp7qA6bOkbMzJsw25WBGPTFa3xL+Bvhb4qfCuf4Qa9e6lb6NPb2tt5lpMEuFWB0dCGKkE/u1z8tej4PcijBHAAxQ20S6MG22tziPhP8J/D/AMH/AIe6R8NPDd1fXGl6PGyQSXcgeZg0ryMXZVAPzP6CvOvjr+xT8Efj9eS674m0e50zxBJH5bavpUggndegWRSCkgxwN6kgcZr3zafT9aXB7mjmafMDoQnDka0R8ZeDv+CW3wP8PaouoeIfE3ifxDCjF1s5pYreJvZjEgcj/dYV9b+HfCHhvwno9l4f8N6Pa6dpunR+Va2ttGI440/ugDt9etbGKKcqjl1FSw9OjpBWCiiipNgr5q8O8ft6+L/+yf6eP/Jtq+la+bPDgz+3n4wJ/wCif6f/AOlT1UepyYneHqfSCnjFPpi9qfUnWFeYftAfG6H4B/Du7+I974T1DXbGwnhiuorKVEeJJH2CQ7+CoYqD/vCvT65X4i+B9K+IngfX/Auugmx12xls5CRnZvQgMv8AtKcMPdRVU+XmTlsRPm5Go7mb8Gvi5ofxq+G+ifEnw7BLBaaxGxNvKQZLaVWKvE5HBYMrDj2p/wAX/i34e+DHw5134meJIpptP0O2MzxQ48yZ8qqRLnjczMoHPU18Sf8ABNvxzq3gbxF8Qf2ZPF7eVqGhX0t9ZxO23EsTCG6RR/d3CFxj+8xq3/wU8+IN94jj8D/s5+E99xqvie/i1C8hj+843+TbJzzgys7Y9Ywa7Hhf9p9mtv0OL60/q3tOtrfM+pf2Z/2jrf8AaU8E3Xj7S/BWp+H9MhvWsoDfzRu1yUAMjrszgAnbzjkN6V6b4j8TaZ4W0LUPEesyiGx0uzlvrqU/djhjQuzH8AfxFcr8FfhfpHwe+Gfhv4daOw8nQbFLd5O80xG6WUgY5Zy5+jGvm7/gpX8XJPCnwnsfhbozu2r+OZxHPFAcuNPhZWk46kO5iQAdQZBXPGEatdQjsbyqSo0HOe9vxOz/AGbf28PBH7Rnj298A6f4R1Lw/eQWkl3bNf3MbfagjKHRVXkMFbd9AfSvp4H1/Cvya+LHwc139ibXfgr8ZPD/AJzzQ2cK65GJiA2oqS9xCecASRSvGMf88WNfqd4b8S6X4q0LTfEei3iXGn6vaRXlnMnSSORQykf8BIz7k1ti6MIWnS2Zlg61SacK26OT+PPxisfgX8LNY+KOoaLPq1to4hMlpBKsbv5kqRjDMMDlx2r4/T/grx4CldNvwd8RFWOMi/gPbJOMZ4r3L/goOV/4ZH8cg5IYWQ64J/02HH1rn/2BNA8Oy/sqeELi80PSrieSS9dpJoY2JH2ubGWK5bgY78Yp0oUlR9pUjd3sOtOq63s4SsrXLXwF/wCCiXwW+OHiC38Im31TwtrV7J5VlBqqp5V0/ZElQlQxPAVtu7I25JxX1H5ynbzgsM4P6V+Xf/BSfQvh9ofxQ8C3vw1tbG08czB5L6LSkRZAVeL7LI6x4xLuLhScMQF64FfpppJvP7Nsm1IgXX2aEz4HSTHzj/vripxNGEIQqR0THhqspzlCTvY8S8Oftf8Ah/xJ+01q37M1t4M1SLUNJE3marJNH9nfy4lkOE+/zuwPevoEdK/Ob4boX/4Kq+JyC+0G/wCBjbxYpX6MFgpINRiacabjbqisLVlVUubozkfil8TNE+FHgDX/AIheIFc2Hh+0e5mRSA0rDhY0z/EzFFHu4rxz9lv9t3wh+0/qmtaJpXhW+8P32kQR3SQ3l1HK1zAzlGdNoGNrbQc8fOvNeDf8FNPibqXiOXwl+zZ4NkFxqGvXcWo6jDGcuxaUxWkJHo0pLEf9M0I4ry7xr4GP7BP7UXw48caJK7+FdV061tNSmVsrIRGlvfE5PXmO4A7Fh6V0UcJGVP3/AImro5quKmq3ufCtz9Xe2R1ryX9o39oXRP2cfAtv461zw7qWs20+oR6eIbJkV1Z1dgxLkcYQ16bZ3kd3apNBIrrJGJUcHKsp5BB7gjn8a+P/APgqbI8H7O+kmOWRN/im1BIx/wA8Ljt3rmw1NTqxpy2OzE1HToynE+jPgz8avCHxz8CWPj7wZO5tbomGe3kKmezuFOGhmVSQrDIPXkFSPvCud/aW/aW8Pfsy+A7fx54l0DUNWt7jU4tMFvZOiurOrtvO8j5QEP5ivz1+CvjTxx+w34y8IeK/E4urz4b/ABQ0u01G5dVZlUOis7qM/wCvgaQ5H8cb+42++f8ABTTU9P8AEf7Mmh654fvre80/UPEdncWdzGwkWaN7echgc4IIIIPp+Arqlg4xrqK1i9jkjjJSoNv4kfYfwz+IVl8TvAHh74g2Fhc2Vr4i0+DUIIJiGeNZV3BWK5XI6da+bvjN/wAFHPA/wb+J+u/DDUvh3ruoXWgvGktzDcQIkheJJPlDHPAfHOM4r139lJF/4Zp+GaEHnw1YZwev7sV6PdeDfCOo3kl9qHhbSLm5lw0k01lG8jkdCWIycYHWuWPs4VJKpG62Or95UpxcZWPjFP8Agq78O3QH/hVHiUsTjAvLbt17+46V7p+zT+1ZoX7SyeIpdC8Iano0fh2W3idryeKTzjKHzjYTjb5Zz1zmvlv9gjRdF1P9qD4029/pNlcwQyXXlwy26Oif8TBhwCMDjiv0H0/Q9H0jd/ZOk2diHwXFtCsQbHTIUDOMnGema3xioUXyQjr3uc2ElXqpTlJW7WNEMDS5FMAx0pHJPU1wI9E+T/jz/wAFB/C3wH+JOp/DXWPhxq+pXGmpA/2qC9hjSXzI0k4VuQBvA69q80sP+Cunw5m1CG21H4S+KLa3lODLHcwSsvJH3CVzntzXJeO0sdQ/4KsaTY6laQ3UDTWgkWcK6tjS2IBBHToRnuBX3F8QvCvwWm8Iavb/ABB0Dwtb6ELOZr2W6ghRI4cbWctj5SM8MOR25r0pQoU4xjyNt+Z5sKlabk1JIufCX41eBPjZ4TtvGngDVBfadcO0LjaVkglUZMciEbkfpwR0IIyCDVH4/wDxw0z4BfDi4+I2q6Hdatb293bWjWtvKkUhMsgQNl+AAT3r4c/4JUvq3/CTfEmy097pvC0ENkYi4AQ3AeQKSMYD+WGGOw2jrzX0H/wUiYr+yrrZDAbtW03PH/Twv+FZSoRhiVT6GkcRKeGdTqefL/wVh+HhGR8JfErA4+7eW5x9cHH612fwV/4KKeDfjT8T9H+GOm/DjXNMudXM226ubmFo08uJ5Oi8nOzH41wP7LX7U/7Jfgr4C+CfCvjrxdodvr2m2TpexTaTJK0chlc7S4iIJ+YcgnpX0t8JvjL+zx8X9bvbL4Satoup6npcX2m4NtprQvFGzFSdzIp5JI49a0rQpwUrU366mdKdSbjeovQ7j4lePLb4cfDvxF8QbnT5b+Dw9p0+oPbxuEaYRoWKhj06dcV8Yx/8Fa/ATsFPwg8Q8sQdt/AcAd8YzjPHrX07+1K3k/s3/EmQOx/4pq/xjg8xt/jXgv8AwTY0TRtX/Zwuri+8P6ddTDxHffvLmJHYYWEgZZc//rrKhTpqi6tSN3cutOo60acJWR0vwV/4KO/BT4weKYPBl3Y6p4U1W9m8iyGqeWYLh+yrKjHDE8AMFByMEk4H1QbpQCwUlVOCfwr8xf8Agqn4b+Huha54G1Hwnptjp/jOdp/tSWESxvJa/JskdUwdwfeFYjPD88DH6EeFNV1LTPhppmteLFIu7TQIrvUTj5hIsIaT8QQ/5UV6MVCE6fXoXQqycpQqbI4n4/ftdfCb9nWGGHxnqM93q92u+10mwQSXUi5xvIJComRje5GeQAxBr50sP+CtXg+4v41vvgt4ht9OkfYLtb2J/wAdpUKf++x9a8t/Y2+GWn/td/H7xn8cPjFaprVlpU0dyljcDdA9zK0nkRup6wwxxEBD8udmQcHP6XXHgnwnd6SdEvvDWk3OmmLyjaS2MbxFP7vlkbce2K0mqGHtTnG766kwlWxCc6crLpocr8HP2gfhn8dvD7a/8O9c+1eRtF1ZTL5d1ascgCSPkgEqQGBKtggHivSAwPXv0r8vviFo1r+wr+2X4d1jwLPLa+D/ABQ8Es2nIx2LaXE3lXEBznKxshlTcTg7B2JP6cOrPDJBFLtd0OxjyFJ6flWFejGDTg/dZrQqymmp/Ej5Z+NX/BRD4afCzxXc+CPDfhjVvGmrWM5t7wWEixwRSA/MgfDM7Kcg7V2ggjdkVQ+Ff/BS74M+Pdfi8L+KtF1jwdfXVwttbPdhbi3eQtgIzxjchJwOVxyOa+Vv2a/jJ4O/ZM+LnjvQvj94M1H+1Lt0sm1RLYTS2YEjb1ZGwzRysVfeh5EanB4NfcvhC+/ZH/aG8U6V4/8ADX/CJeI/Efh2ZZ7W4UCK+gccK0kR2yNjO5fMUhTtIwenTUo0qStyN/3rnLTrVakrqaX92x7/AOb0wjfjxXgXgD9sHwz4/wD2h/En7PFl4R1a11Lw39pEuoztH5E3kMobaud3O7j9a91MhAj278t2PP4c1+df7OKo/wDwUz+J24FlH9snaRx/roR0/E1yUKanGbfQ6q9SUZQS6n2P+0d+0Non7N3w9X4ieIPDuoavaNfwWBgs3RXDSBiGyxAx8pr5ih/4K2/DO4Py/CvxOBll3G7th0/4F9OuK+5L3RtN1KJoNRsbe6idg5iniWRAw6HBBGa/Pj/gqv4f0jRvDvw+k0jS7CyM1/qPmG3gSPf+6h67QM1vgo0aslTnHV+Zni5VqMXUjLT0O28Mf8FUvhp4i17S/D8Hw08TQTarqFvp8bvc2pVWldVViQ/T5x+VfbZu1jfy3RskEg4zwPp/LvXLeEfCfhVfDuhXKeG9JWQWNvIHWyiDB/LX5gQuc+9ebftk/EzVfhN+z14l8SaDcPaavfmPS7KZPlaJpmIZwexEYlKnsQuKymqdWoo0o26b3NISqUqbnVlf5WOX+N//AAUE+Dvwg1S58N6ba6j4v1q0fyrmHS9gtreT+5JcMdu7sQgfHfB4rgfCX/BU/wCHd/qn2Hx38N/EHhm23Afa0lS7VVJ4dl2owX1KhuO1O/4J+fs2+B9O+Gdn8ZfFHh+z1LxN4lllmsZLxFmWzt0kZAYlYELI7Kzs45wVAIwc/S/xY+CPw3+MXhm88M+NvDtlMLmIrb3aQqtxaPjAkikxuVlPOM4PQ5FbylhKUnS5W2utzCLxFSKqcyV+ljqPCvjTw7420Oy8S+E9XttU0rUF3wXlvIGjYf054x1ByCMiuc+Onxetfgj8L9a+J17ok+qQaKIi9pFIsTyB5UThjkD7+enavhr/AIJ4eKvEfw0+Ovjf9mHXdQkltLWW7uLZS3yx3dpII5XjXt5kZLHHXy19TX0/+3cw/wCGTvHbqhO6Ky3YO3P+mQVlKgoV/Z9DWNdzoOb0aPBl/wCCu/gLAL/BrxIAQMEXsLcnsRjI/WvWPgx/wUR+DHxe1y38MXFjq3hXU7yYQWi6miGG4cjIRZUJAc9lbbnjGTxVL9gDwvoN9+y54UlvfD+l3Exn1BzLPAjsf9KmA5K5+mT0AxXzz/wVD0f4c6N4r8FXPgyz0208at9obUE09Y438vMXkySKgyHB8wIT8xAOeADXQqVCrV9gotedzmdWvTp+25rrtY/STXNau9N0i+1DTNLn1S6tLV7iKyiwj3LBSVjRmwNzEbRnjJ5ryD9nL9rnwP8AtHS63puiaVf6FrGhSL5+m6iU894W4EoCk8BgysOqnb1yM+q+D/7TfwdoMniCNk1I6Vbm9QnnzjEpkz64bcK+AP2vPB2v/sp/tA+H/wBqf4bWjLomt3ZTWbKP5Y/tcg/fRMOy3EYZge0iMeMLXNQpwqt05b9Drr1J0+Wa2Pur4o/Fzwn8IvAuq/EDxddNFpmlQmRgpXzJnztWKNSeXZyqKDjk+gJrlf2c/wBoVf2iPCd34407wJqegaRHeGztJb6eNmuyoG9lVeiqx2k85YHHSviD9pPx7qH7b/xw8D/AT4QanKfCkaw6vfXSglI3eINLM4/6YRPsA/56yOvcGv0Z8DeA/Dnw88K6R4N8K2f2PS9Eto7S2gU9FUdT6sSSzHqSSe5zVWjChTXN8T/BCpVZV6j5fhR09FFFch1BXzZ4Z5/by8Y57fD/AE7/ANKnr6Tr5r8MsP8AhvTxkuP+af6d/wClUlXDqcuI0cPVH0iOKfTBzT6g6gqOWEMv3sc5/GpKDyMGgD8zf2yLe7/Zo/bJ8EftHaDbNHpniSRP7USPCq0kYWG6Q9vnhZHH+2rE1P8As02DftSft0+L/j1d5n8L+CXK6W7n5WZQ0NpgHpwss/1Kk19iftM/s1+G/wBpb4er4E17V59JeC+S+s9QghSSSFwCrDaxG5WViDyOo9KX9mT9m3w1+zL4Al8EeH9Wn1aS6vGvbvUJ4liknfAVVKqSAqqoABJ6n1r0frcfq/8Aftb5HmLCT9vb7F7nrvJ4YvnGWGfu5PH8sV+THxZ8a/FD9oP9rPVPH3wc8GyeNU8DXUKaXa/ZvtFsttbSFY3cB1yrz+bJjcpPHYYr9TPHPhq78WeEta8Mafr9xolzq9lJaDULZQ8tsHUqXVWONwBP8+teafs1/steFP2bPD+raNousXWs3Ws3aXFzqF3CkcxRECxxApn5V+Yj3ZjWWDrQwqlN6y2X6muLoTxLjT+z1Pi743+J/wBvT42+AJ/APjr9m8GymnS7S4s9MkWe2kiPyshNwwBILK2Qcq7Yr2P/AIJmfGdfFXwtvfhRrNwx1TwPODaI6kO2nzMSvHX93L5iY7AxivtV7eJySy84wPpXzb8PP2JPD3ws/aB1X47eCvHep2K6zNdPdaCttGbVorg7njDZ3ACTa68cFRWksVTrUXSkrdVYhYWpRrKonfo7i/8ABQ3I/ZH8YqhwzNp52kYOPtkJ/wDrV8gfs+fsHxfGX4EaZ8S9L+K2saTqWorexw6clpvgEiSvGAZA4badgJOOCxr9Efjt8H7T44fC7VPhffa7eaRBqjwSNd26q8ieVMso+ViOpTFN+Bnwcsfgh8LtJ+GFjrtzq8OkyTOt5NGscshkmaQkqCehbFTSxnsaHLB63uVWw3t8RzTXu2sfnP8AsGeCvhton7Q9/wCDvjpo11B8QNFuiNDjvJ1Nst5FkMjLtG6dQC8bFijLnAztJ/VQsybXY/MzYI2/59q+a/2if2HvDnx0+ImlfFLRfG+o+CvE2nbPNvdOtlla4MZzDKfmXa6YwG9l9Ofo7SLO/stJsrPVtTN/eQW6RT3QiEPnyAAM+wEhdxBO0E47VGKrRryVRfd0+ReDpTop02tL6Pr8z85/hjeTH/gq74njcYUyaiBxjP8AoKf4V+jGravp+k6VfavqVytvZ6fbyXNxO/CxxIu52PsACT+NeB+Hv2ONN8PftRah+05H49vprzUJLl20l7OPykEsPlYEm7PA5+7XpXxx+F2pfGD4Z6z8ObDxlceG11pUhub2K2WdjDuBkQKWXhx8p56Z9TU1pwqzjrskFCnOjGWm7Z+WXh7W/j38a/2jtd/aR+Enw2fxTd6dqhkt47m3EsFmjKY7VCpkUFkgTIAIwx3e57r9pOb9uL44eBotL+KPwAgttP0Gd9VF7ZWe2SBURhLlvPb5DGTuG3kgY5Ar9AP2ePgB4b/Z5+HMXgLRr5tSla5lvL7UngWGS7mc/eZVJACqqoBk8IK9MutOtbyCa2uIkeG4Ro5UYZDq3UH1BrqePhCouWC02OaOBnODcptN7nzb+wB8Y4Pir+z3pFpd3fn6x4Pb+wr4E5Zo41BgkPc5h25PdkeuK/4KqTqv7POkbE3Mviu14AP/AD73Fehfs4fsa6R+zX4z8ReJvCnxA1i90/xFEYptIubeMRRBZC0JDj5iyAsoOPmBNdL+05+zdY/tK/D228A3/iq70GO21KLUBdQQidmKJImzaxH/AD0/vdqxVSlHE+0jtc39lUlhfZy+Kxw2hfBPwr8fv2NfAXw98VQbPtHhjT5rG9jQtLY3S26mKVeByvdf4lLKcZDD83fiZ4v+J/wt+Her/sdfFDTpbn/hH/EMOq6Td7yyRRKku4RseWikEgljI+6fMHGSB+yvw78BQeAPh54e8AR6jJfJ4e0uDTFunQI0oijCB9oJCk4PGT1615N+0/8AsZ+CP2l7XTbm+1Wfw/4g0uQLFq9vAsryQZLGF0LLuG7DA5yuT0yc6YbFxp1HGp8N7ryM8Rg5Tppw+K1n5nVfsnEH9mv4ZAkZ/wCEasOmP+eYr15ev4VyXwt8BQfDL4d+HPAFvqLahH4c06DTlu3jEbTLGuA5QEhc49a63pXDUkpzcl1Z304uEFF9Efnd/wAE8g3/AA1F8bCVYBjdHkd/7Rev0TwK8C+A/wCyboXwK+IXiz4haT4x1TVJ/Fnmefa3MMarDvnM3ysvJ5JFe/Vpi6sa1Xmh2X5GGEpypUlGXd/mN2n1qNqmqNk6+1c6VjqPyc/aU+H4+LP/AAUef4eTarc6PHrk9lENQgTdJFt08NwuQG+7t696wP2uv2NfEv7POi6P4xtvGOo+MfDE90bW/mnj8p7eTeSi9XADgMu/s64IIYV99av+xx4d1X9pq1/aZPjbWIdUtZ4Zv7LWKI2zbIPIC7vvcjnrXsfjv4feGviL4P1PwP4rsFvdK1eFobmJyMsG7qf4WBAYEchgMV6n19xcEvhSV0eV9R54zvu27HlX7ImmfBPT/gxoN18CrVodD1ZGnuTNIZbs3YG2RbhgP9arAqeNo24UAHni/wDgpMk037KmsCGOQkatpuQR6XAzXS/st/slXP7Ls2s6foXxT1LXPDusOJxpN5YxotvOOFlR1Y/MUG1vl+banTFd7+0N8ErL9oH4Y3vw0vtem0WG8uoLn7XBAJmUxvvA2kgc/WuXnhHEqad1c6eScsO42s7HzV+y3+yB+zZ4y/Z+8CeK/GHwz0y91vVdMSa8nlu5ozM+9wSVDgDoOAuK+kvhb8Dfgr8IL+61T4Z+D9N0S91SEW0rQXDu0sY+YDDOeh9K+WR/wSg0FYlhT49+JVjWPYEXToguPTG+uw+Dn/BO3T/hB8TdC+Itt8Zda1V9DnM32OaxjRZsxMhVmEhI+9np2FbVZ05p/vXZ9LGNGE4NfutV1ue0ftYuV/Zq+JZwSB4avB6/8szX57fss/sifEz43/DKfxf4V+OV74Q07+1rqybToRcEb0C7m/dzIuSHXjGfkHNfpz8U/h7D8UPh14h+H13qM1hD4i0+awkuYEDPGrjG4A8dK5P9nD9n6x/Z18CP4E0zxPd65BJqE2oG4uoFjfdIqKVwpPA2A596ihiFQoOMN2y62HdeunPZI8J+EP8AwTW8F+DfGFn4/wDiZ481Px1qdhKs8UE8HlWxmQgq8gLu0m3A+UkKcDINfXmuaPBqui32jzgmG/tJLSXk5CupU4/76NaagZBAGPT0pSM9a5qtWdaSnUd7HVSoQopqnpc/L79hbx5a/swfHrxv8CvirMmix6rOltDeXLCKH7VAzCMljwElWRiGJx/q843cfpzLfWUVob6a5hS2VPNMzOAgTGd2c4xjv0714Z+0P+x18Kv2io01LxBFcaT4jt4/Lg1uw2iYL2WVT8sqDsDhh2YV85r/AMEv/G8wfRb/APaV1KTw8z5NktjKRs/65Gfy/wBOvNdVSWHxT9rKXK3uclOOIw/ucvMuh5/+0L4htP2wv2z/AAl4E+HEjahoWgS29lPf2w3RPHHMZbudSOsYXMYblSyrg/Otfpvc3dtY2NzfXt1FaW1rE0s07sFSNFBZnYnhQACST2FeW/s//ssfC/8AZ00p7bwRZzT6pc4F9q97IJLq5/2CQMKg6hFAGcE5Nep6rommazpF3oms2cV9ZX8D2t1BMuUmicFWRh3BBIrOvVhPlhD4YrQ1w1KdPmlP4m9Txi80/wDZZ/a60C5b7L4X8ci2JX5cx31uF46/LPGpPTGFI6dq/P79s34C+Ev2S/GPhDxd8FvGWp6dqV/NJcxaa94JJ7Jo9pE0Mg+Yx7uPmyevJywP0f8AEP8A4Jf+G7zXbnX/AIP/ABS1XwY8+4pZzRG5jiJOSI5FkSQKeMgsx96n+Fv/AATM8OaF4ztfHfxe+Jeo+O72zljlS1khaOGZ4zlDK7ySPKo/u5XjI6cV00qtGk+ZTuuxzVKdar7vJZ90fXvhO+1DUvC+jatq8QS9vNPtp7iJWx5crRgyDHoGJH4V8E/s4gr/AMFL/ic5Vl3f217jHnRc1+ivkgDgYOAM+3p9K+f/AAB+yDofgD9onxD+0JZ+NtSur3xD9t8/TJYYxFH9odWOHHzcbfSuahUjFTv1R0VqU5ODXRn0Jt7Zr8+P+Ct8bHwp8OyqOSL/AFDp/wBcoR/j+VfoRXg/7U37KeiftRafoGm614w1HQF0CaeZGtIkl80yhAQd+MY2frU4OoqNZTlsjTFwdWi4x6nrHgiUSeD/AA+5UDdpdow+b1iWvFv25vhtqnxN/Zt8SaRocDTahprR6vEiglpBBkuAO58oykAdWAxXvOiaSmi6Np+jpI0q2FpFarIerBFVQf0q08CkNu+dSSCD6Gs41PZ1FUXRmkqanT9m+x8X/wDBOr9ojwj4m+EOm/CPVdTtbHxV4XZ7aGzuZQpvLV2Z0eMZ+baG2EDnCqcYYV9S/Ef4oeDPhV4WvfGXjjXbfS9Os1c/PIu+dwP9XEucyOeygEmvmv42f8E2vhf8SNeufGHgHxBe+A9aupWuJktIRNYyS8ZcRZUxscDlGA/2a4TQP+CVkN/q0V98WPjrrXiC1glB8i2iaN2X+JfNlkkKA+wrsmsNWl7Xntd66HHB4mlH2fLfQ4/9gLRtd+K37T/jb9orUNPNrYp9uIfqDc3km9Yg2cHbFu6dihP3sj6t/bpAP7KXjllTcxhs22EHn/TIK9a8AfDHwd8MPDVh4R8C6PFpGk6fuMcEOfnLfeZySSzE8liSSQCTxWP8cfhUnxp+FusfDKXXbjSItWWFDeQxCZ49kqSfdYgNygHUdaiddTxEZr4Vb7i4UJU8O47t3Z+eX7Of7FEPxm+Clp8SNN+LGt6FqFzPfW8WnRQb4BJFK8YLMHViDgE8cZNUv2Mfh18PLD9oq98EfH3TryLxvoF6TotvdShrSW6iPKuu3Mkq48yMltjrzg4UH9DfgD8FLb4E/C2w+GVvr02tR2M89x9tlhEDOZJS+NqscYJx1rzj9on9ijw38dfG+jfEbS/GOoeDPEel7fNvtOgV3uRGQYpG5UiRMcPk8YB4Arp+vKcp05Oyd7Pqjm+puEITirtWuj6NDEn+HcCq4z9Mjn2ryv8Aai8LaH4s+APj3SPENmbm0/sO9vUwPmjnhjaWJwexVo0I+lekaJYahYaXaWOq6mdQu7eFIp7oxiMzOBhn2KSF3Hn27VS8eeD7fx54N1vwZdXs1pBrljcWEk0IBeJZomjZlB4yA2cGvMhLkmpdj1Jx54NHwx/wSW8KaB/whnjPxq9iv9vS6rFpn2hsZW0EKyiNRjgMzsSO+1M/dr9CsccV4p+zH+zJoX7MfhjVfC/h/wATX+tQ6rqI1CSW9hjV0YRrHtG3HHy9fevbMA1ti6vt6zl0ZhhabpUlF79QooornOkK+aPDTY/b38ZDH/NPtN/9K3/xr6WFfNHh0/8AGfXjFf8Aqnum/wDpW1XDqcmK3g/NH0qtPpi9qfUHWFFFYniXxj4e8G6TLrvinU4tN0+Eqr3E33AWbaoyMnknjigHpqbW0enWm4XqRnHqa8vk/ak/Z8i/1nxZ0Bf96cj+lQD9qz9nVmwvxf8ADh/7ef8A61VyO9zL6xS25kesDAGAOPTNJgf3jXlX/DVH7PP/AEVvw9/3/P8AhSH9qf8AZ6H/ADVzQP8Av+f8KXI97C9vS25kerUuT615P/w1V+z1/wBFY0H/AL/t/wDE0/8A4am/Z6/6K14f/wC/5/wp8kuwfWKX86PVc0fL0Jryv/hqb9nr/orWgf8Af8/4Uz/hqf8AZ7/6Kz4f/wC/5/wpcj3sH1iltzI9W2r3P4dqUkmvJ/8Ahqn9nn/oregf9/j/AIU8ftTfs9Hp8WdA/wC/5/wp8jWyD6xR/mR6rkelJXlv/DUn7Pf/AEVnQP8Av+f8KQ/tSfs9j/mrXh//AMCD/hS5HvYPrFLX3keqAY43GjA/vfpXlP8Aw1L+z7/0VnQf+/x/woX9qX9nw/8ANWtB/wC/x/wquV9g+s0v5keq7VxjPFGBnOK8r/4ak/Z9/wCiteH/APwI/wDrUv8Aw1J+z5/0Vjw//wCBH/1qXI9rB9Zpb8yPUwpByDz60gVcYIGCMH3ry3/hqT9nz/orHh//AMCP/rU9f2n/ANn4/wDNWPD3r/x9f/WqeRvQFiKW/Oj1HIJzjmkrzL/hpr4A/wDRV9A/8Cf/AK1Rt+1D+z8vX4seHx/28H/Cn7Nj9vS/mR6l3zS4Ydq8n/4ao/Z9HB+LPh8/9tz/AIUv/DVX7PX/AEVnw/8A9/z/AIUuXyB4il/Mj1jDelMPNeW/8NUfs+f9FZ0D/v8An/Cmf8NUfs8/9Fb8P/8Af8/4U7MX1il/Mj1XaOmKXJIxXlq/tQfs/v8Ac+K2gH/tuf8ACpD+0z8BAMn4p6H+MxH9KEpdR/WKT+0j0zaep203eR94A84+U0kMsVxEs8LbkYZDV8IfGP8Ab98baB411LSPh74e0n+y9I1B7FpL6NnkuGjcq74DLtUkHAGTjHOTgSlZWObG4+lgYqdV7n3luPfH/fVG3pwOOleb/AT4uWnxr+Gum+PLKz+ytdNJBPb8sEmjYq21sfMpIyD6EVd8Y/HT4VfD7VTonjTxrp2k33libybksrbGJCtwpGMhh/wE0RUmzohVhOmq19Gd5160V5I37WX7O4/5qton/fb/APxNPX9q39nk/wDNVdF/77f/AOJq/Zy3sHt6VrcyPWBxRXlo/aj/AGfW6fFTRT/wN/8A4mj/AIai/Z//AOiqaF/39b/ClysPb0v5kepYHf8AnSBACSCMnqcV5h/w0/8AAH/oqeif9/G/+JpH/ah+AC9fitoI+sp/wpcsuoe3o/zL7z1LanSg5IxgfnXk/wDw1P8As+9vivoX/f5v/iaaf2rP2fF6/FfQf+/zf/E1Xs5dhPEUX9pHq2Oc5P50Y5zkn6nNeVL+1P8As+O3y/FjQP8Av83/AMTU4/ac+AB6/FfQP/Ag/wCFHLLcf1ija3MvvPUNrf3qMH0XivMT+09+z+OnxW0D/wACD/hTH/ai/Z/jHz/Fnw8P+25/wo5X2H7el/Mj1KgAA5Ax9K8qH7VH7PJ6fFnQD/23P+FB/an/AGe1+98WvD4/7eD/AIUOnJ9BLE0v5kerceh/OlyvqK8nH7VH7PTdPizoB/7bH/CpE/ai/Z+c4X4saAf+25/wo9nJ9A+sUv5kepdKaI0BBAwR05rzQftNfAEdfit4f/G5x/Sl/wCGmPgH/wBFV0D/AMCf/rU+SV76gq9K1uZfeem9ORxSbPcV5i37TfwCUc/FbQB/23P+FV2/an/Z8U8/FXQP/Ag//E0ezn2D6zSj9pHqvPpShFHQj8q8m/4ap/Z9/wCiseH/APwIP+FTx/tRfAGQZHxY0D/v/ScG1axKxNJfbX3nqZUHgsSKPl/u9eteVD9qP4AFufix4f8A+/8A/wDWqX/hp34A9/itoA/7bn/CjldivrFK/wASPUdoHalrzH/hpv4BY/5KroP/AH/P+FdL4L+J3gT4iC7PgnxTY6x9iKCf7M+7y92Su7gcnaaXK0WqkZaJnU0UUUixB/Svmfw82P2/fF4x1+Hmn/8ApY1fTPevmTw9z+394t/7J3p//pa1XDr6HJil8Hqj6YXtT6YvXFPqDrCq9xYWV3GYru2jnjYhikq7xkHI4PpViojMqnBUkY4xzmgVk1qVW8N6A3XRNP8A/AZP8KYPDPhwHI0PT/8AwGT/AAq99oPBK4B75GMY/rTgxzg9/Si7QcqKH/COeHv+gDp3/gKn+FH/AAjnh4/8wHTv/AVP8K0aKLsOVGX/AMIx4d/6AOnf+Asf+FL/AMI14f8A+gHp/wD4DJ/hWmeKQtgkenr9aLyDlRS/4Rzw9/0AtP8A/AZP8Kj/AOEb0D/oB6f/AOAqf4VfMxGclRg4zngfjQsgYgKwIPPX8fyouxWRR/4Rzw9/0AtO/wDAVP8ACl/4Rvw//wBALT//AAFj/wAK0M5pGlC7sk/KM8Ci77hyoof8I74f/wCgFp//AICp/hSHw54ePXQdO/8AAWP/AAq155yRtJ2nk+x9uuc8YqYMD/EKLsfKjO/4Rrw5/wBADTv/AAFT/CgeGvDg6aBpv/gLH/hWlggZP60A/Si7DlRm/wDCNeHf+gDp3/gKn+FH/CNeHf8AoA6d/wCAqf4Vo59ePTPejJxyvIPai7FyIzv+Ea8O/wDQB07/AMBU/wAKX/hGfD3/AEAdP/8AAWP/AAq+WYEccH/GlLkEjHOM4p3YcqKP/CP6D/0BLD/wGT/Cmnw74fbroenn/t2T/CrplxuAx8uBnPfpj60GUbiBggde3v8Al7/Skmx2RTHhzw8P+YJp/wD4Cp/hR/wjvh//AKAOnf8AgLH/AIVe8xccHn6U0ylQSwCgY6nt/wDqpajsil/wjvh//oB2H/gMn+FN/wCEZ8P/APQE0/8A8BU/wrQWRWOAw6A9R36Cn9Bk0WfcVkZ6eH9DT/mD2P8A4DJ/hQ/h7w+4+bRdPb62qH+lXw2SfY49803dg7c+/SquwshFiVV2JhVAwABgV8Y/Fn/gntD418eX3i/wh45i0O01W7N5cWdxZmXy3cln2sGXKFmbCnBAOM44r7SPA/xqIHnJZQTnHFK9zmxODpYtKNVXscR8IfhR4f8Ag54C0/wJoJM9vZhneeZQGmlY5aRgOhJ6Y6V1txouj3cvm3emWk0uMb5IFZvzI9z+ZqwGXpkN16e5x/X/APXU+ARnNO7vc3hShTgoRWiWxnf8I7oR/wCYNZf+A6f4Uo8O6EOmkWP/AIDp/hWhuX1qHzQeVC47E9z6fn3ovLuVZEA0PRR00iy/8B0/wp/9jaT/ANAy0/78r/hUpmGV+YDJ6E47Z4/z61K0ihd2cgelK77jsip/Yej/APQKs/8Avwn+FMOhaI33tHsj/wBu6f4VaE+WZMj5RnOeMU8njggmndi5UZq+GtBX/mC6ef8At1T/AApH8MeHX66Hp/8A4Cp/hWmTyBx+dIHJyNuCO2eetK8u4uRGcnhnw9GcjQ9P/wDAVP8ACpP7A0Af8wbTv/AdP8KtCbGCduD0O72zQbhRtDbAWYqMt3/zk/gaLvuPlRUPh/QT/wAwWw/8Bk/woPh3w+eDoenn/t2T/Cr+/Hp1x1pA+ei/TnvTu+4WRQXw14eXpoenf+Aqf4UP4Z8Ov10PT/8AwFT/AAq/5ueAMnGcUhmIGffn2pXkLkRRHhrw8OmiWH/gKn+FKPDugA5GiWH/AIDJ/hVzzhnG5R/wLnPX+Rp+7nFF5dw5EUT4f0E/e0SwP1tkP9KP7A0L/oC2H/gMn+FXS4Bx6jOfp1piy7hkFeuPvf8A1uvt9KLsrTsVD4f8PkYOhaf/AOAqf4VC3hPww3P/AAj2mf8AgHH/AIVph8kDoT2Ppx/jT+/40JvuTyoxf+EQ8L/9C/pv/gJH/hTl8J+GUPy6DpoH/XpH/hWxTd67tpPPFLXuHJHsvuM3/hF/Dg5/sHTv/AWP/CpP+Ed8P/8AQC0//wABU/wq2JgxABXn/aGfy7ipsGndj5UZ3/CPaB0/sSw/8Bk/wqS10bSbBmex0y0t2cgsYoVTJAwM4FXPQU7IptsSik9ENooopFB3r5j8PMF/b/8AFgPyk/DrTz8wxz9tcY/Svpvd7V5xY/BvTdP+OWo/G+HW7z7bqOgQaBJYFV8kRxTtKJA2N24lsdaqLtcwrwc+W3Rnoo4O4U+mdsU7d7VJuLXmHx/8c618OfhNr3inw59nGqwiC0sXmGYYp7m4jt4pH/2VeUMR3CnpXp272rn/ABp4L8PePvCmq+C/FNmbrS9atntbmPfsYo3OVYcqykgggggqCOeaFo9QOH8LfA+bwtf6f4im+Lvj7UNXtJBNqk19qzPa6iNn7xGtmUwxIScjylVlxgNwc8pZftK+KptI0/4nXPwxjh+FmpX8VnBrI1YHUUgmnEEN9JaeXtEDyMnAlLqjhip6V0nh/wCEXxV0vUdJi1v9ofW9Z0DRnBSwfRbSK5vUUELHd3SgtMuMb9qIXxknJNc7Z/sw6xBp1h4Buvi3qN18NdKv47y18ONpsIn8uKYTw2sl7nc9ukgXC7QxVFUueavR7kEGtftQeM9HHibxH/wqVZfB/g3xPL4e1fU/7YVbhkWeKI3EFt5fzqnmqXDSKeu3dV3xj+0l4u0LVfH8Ph/4TnVtK+Ghhn1y/l1pLZpLZ7VLh/s0Xlt5kqIznazIp2Ab8tgbGq/s92+rfDvxt8P38SvGvjTxDNr8l19kGbUyXMU3lBQ+Gx5W3cSOvSrmq/A+21C2+LVu+vPH/wALUtxbzN9mB+wAWC2gK/P+86b/AOH06c0rodmc/wCNf2orTRvEtz4a8IaToeoy6Xp9rqV+2seJbbSCRcR+bHb26ygmaUxFWJO2Mb0BcEkCxaftEah4+nsLT4G+C4vFRm0K28RXk19qa2EFrBcFxBAXCSFrhzFKCm0IuzJfkZg8R/s33Nx4jl8R+EfE+j2NzqWmWWnaomr+GoNVSRraLyormESMrQy7AoIJaM7VypIJOhqXwN8S6brtv4q+F/xGXw1qsujW2jaw02iw3VrqMdvvaGcW6GJIplaWbDJ8mHxsIAAegrswdF8b/FnVf2iPEnhXVPDljaeGoPCum389u2rbprYyfagZFVIiGkaSPyyN+AsSuCd20Y/w8+MniZ/D3gTwL8J/hct/PqvglPEcP9s+JH2WMAlSLyprhonklY7uGC5J6hRzXplx8KPEkPxEHxA0Tx8lvc3uh2uia5DdaTHOL5bdpWjljZXj8iTdPIT8rIcrhRiqfwz+Adp8Nb/w1qEXiafUJPDnhFPCS7rQRCaNZ1l89sFtp+UDb05PNHugk0c7oP7Snir4hnS7X4U/DCPVL/8AsSHW9ct9T1gWS2HmyzQraJIsUglmMltcYOFTEYJYbgKnn/aHv739mDUP2grDw2bK9XSrq4t9LuJtyxXCTPBGsrAfd3qNxHTBHvXnvif4U6r8L9S020+Hy/FRb4aVLps+reF7TTp4dRt3u7i4W1nFw2LeSN53KT4ACyEAscV658Gvg9D4Y/Z30P4N+PbKG/jbSp7PVbYymRXEzO8se8YLY8wrvGCSMiiSiGp5/wCIbTw18INQ0XWfij+0544svFc0UN9dXLq7aPcpvAki+zJA1vHETlQuRIAQ249Tt+OP2s9G8I+J/EVlaaXo13ovg6eO21qe48RW9pqDv5ayTGzsnG64ESOuSWTc4dEDFcm1qP7PvxF1vQ0+HWufHnU77wKESCWzfR7f+1Li0UjFvLf5ywIUKXEQdhncxyc2tZ/Z91hPGGt+IfAvjaw0Ky8U3kV9qltceHbe/mjuQiK8lrLIf3W9Y0JVlkUNuYAZOVo9wKPiv9prxDot/wCNZvD/AMLJNZ8O+AGt5tY1QavHEWtJLWK5aSCFkzIyxyFtpZQQB82SVGzrHxz8TX3ibVfDnwp+GreLv7Atra51e5m1WOwjRriITRW8O5XMs3lFXIOxRvQFgTirmr/Amz1TS/inpja7LGvxPiEcrLbqPsOLJLQFADiTGwPzt64rF1P9n7xdp2vXfiD4X/F258JT65p9nZa7EdKgvEuXt4vKiuYhIR5M3ljaT8wOBlSeaLoPeKL/ALT2peJL3Q7H4U/DWbxJJ4g8MTeJIvtmpJp4tVinELwzblciQOduBn5hjgAvUngL9pXXPGkvgnWtT+GFxonhX4hzNZ6PqEmppLcpdCGSYLPbqo8tHEMoDCRj8oyF3V0vgr9n3wz4B1vQ9T0DULk2+g+FpvDMdvOodphLcieS4kk6mRpAxPHJdjUXh/4AaboHhD4beD4fENxInw41FdRgma3UG9ZYbiHa65+UEXBORn7tF0OzOR8N/tg+H/EGt6Nu0/RovDviLVxo+nTx+I7aXVFlkkMcEs+nj54opGA/iZ1DqWVcnb0Hwe+P9x8XdWvltND0m10q3NwoVdcjl1S2aOUxhbyy2B7cttLAbmABGTUfhD9njUPBOp2dronjqJPC2mXclzY6W2g2jXUSElkt2vWBdoUY5HyiTaqjzODVFf2bfEdz4zX4har8VftXiPTtOvrHSdSh8OWdtPC90oUy3JQYutigFUYKueSM807RF7xt/tQ+Idb8Ofs+eM/FHhvUpdN1LT9OEtvcwHDxP5qDcCOnem/F3xDqFlbfDCax1G4tmvvGmk2twYZvL86F4pi6Nj7y8DKnrgdK7rW/AuneLvAN14A8ZzNq1pqWm/2dqMrL5LXG6PbI+FxsJ+9x0PTFeZ+GP2ePFdrrvhqXxz8a9a8VaF4NuI7zRNLnsLe3YTxo0cMlzPH81wY0Y4+7ljk5pKxWpSg/aQ8ZlbjxTdfC+2h8GWHi6Xwpd6h/bObwML77Glylv5WDHvKblLhuSQCACYPi/wDFvxtrfgz4nW3gL4fjUPD/AIYsNS0jUNXbVhbXRultmMr2kGw+YsG/5maRNxVgmSK6q5+AFtP8ONS+Hq+Jpkj1LxU/idrr7IuVZtSF95OzdgjI2bs5xzjPFZPjD9nLxLrE/jDTPCPxg1Hwx4a8cyXF1q+lxaZBO32meIJO8MzndEsmAzrg5O7BXNPQk5e6/aVl8IQ6X4U0PT/Dt/daD4Z0u+1Z9c8UQaXJO81urrDbCUEzSbCGLEqo8xAWBOR1vh/9pJvGniXTNK8DeAr/AFfRr3w1Y+K5tU+0rGLeyuDMNpjwS8wMIURoTuJY7lCZanqv7Mt2NSGseD/HFrpl9d6TYaXqzX/h621FLlrWHyYrmIS4MEmz5TguhwpKkqSe50T4Tx+HfE1/4n03xBOl3eeGbPw6rC3iUxm3edxcABdhctOTtCBfl6UtCtTyTVP2m/Gniz9n3xx8TfBPg3T7WfStFe/tJbfxJaXbWqmNmY3CKpMVxAo3tAyHcRsD5zXVXfxq8f8AhX4YeGPEviXwZoFnq2pQ7riHVPGFrZQBFjBDid4wGeTcpCKmFz8zCqGn/sqNe3PjTUvHfjsavqfjLw5J4ZuLvT9Gg0xmt3yWnmWLInuMniRsADICgEgrP+zb4uvr3wv4h1T4oWF34g0HSpdDe6uPDNvPC1ozo6tFBI5WGddmDKCwYHBUjinaJJYsv2mdT8Wy+E7P4X/DSXxBceLPDD+JIhdavFZx2saTRwtHK+1wSHkAygYEjoFO4TaT+0TqfjeDw/afDH4czazrWsaNJrl/Z3+ppYxaZAkzW5jkm2SbpWnjlRAqkHyyxKrzV/4X/s82vwy1LwpqMPii41I+F/DV14bj8y2EfnpLdLcecxBIDDbtwBhsk8fdrJ0r9mzWvBkWh3vw3+KFxoWt6XYXWk3F5LpUV1Hf2Ut3JdBHhc/K8ckrbHVhgEggg4otHoHvHFfBfx9a38vhG68Q6Nq51S/1vx1ew+ZqjoLQQ3Tb4JYgSk52uEX5sIVyM547XwR+0vruv/8ACFat4s+F83hzw14/jYaRqT6tHcOswtnuAs8SoNiOkUhRwzZABZV3ACx4L/Zlj8I/2FJL4/v9Uk0N/EMgmubSMPcHVXDOXKsBujYMeBhtx+7W3H8CNKt/CPw28LNrk8sXw1mtriGQwpm+8m0ltgsg6KCspb5c8jHSn7oanHH9qbxV/ZvhbxafgteJ4T8aa3YaRo+pvrMKzeXdzBIp57bZuiVkJdVDOT8qsULZD/2q9fbQ7/4ZWd94j8V6VomoeJbiDVT4Ze6F7JCLC5ZFC2oMzASqhIVTwDXj2mfDzxrr3iDwT8NtD074q2GieEPFdpqyaZ4gtbKLStIsraVpCq3cXz3uRtSBdxCBgWyQCPpn4ufCrW/iLN4V1bw546bwtqvhPUpdStLs6bHeqzSW0luVaOQhcbJXNOyuK0nueF+D/jsPh5pXxP1bTb7xV4x8O+G73RU0mw8RzPFrccl2whuAFnQXH2fJjaPzVG5hIoOOa9V8cfHDx/4F8Mabq+ufDnw9pl/frdzTWmreM7WzjgSI/uo1lZD50zryVRdi8guetZsn7L9/r0Wv6h4/+KV/4i8ReIH0mKXUv7MhtYre00+8F1FbxW6HCgyByzEkkv8AStb4ofs7t8QviAPH+meNf7HuZ9HGhXkU+jW+oj7MJXfdbtNxbS5kcFwDnjIOBhe6GpxV5+0N4cm1C0+LWneHNYlz8Kb3xhFA2qGKMwJPEWt3gAKebu/5bEkgcYwSK3bf9pvxDp1wzeOPhDqejQX/AIZvPFGiJa6jFfXF7BaxJJLA8SAeXPiSMhdzD5sEgg0/T/2TdHsvCkPhYeMr144Ph/c+ARN9lQEwTSh/tGN2A4wAF6e9dhr/AMHJdV1bwvrmkeMLnSdQ8J6LfaNZzRWkc2WuI4UExV8rlPJU7cYO45o9wfvDvgx8VNT+K/hyTxNdaZ4ft7OQRG0k0jX49UDFxkpKUjTy5FyoKnuTWz8U/H9p8Mfh5rfje7USPptqxt4jx9oumIWGIH/alZFz2ySeK840H9nfxrodxruvQfF97PxD4hvNOa9v9K0C2so2tbRnby/IyytJKJGDyMSSNoxgV3fxO+FGl/Fabw7ZeJ7xZtE0PUxql5pL2yyRak6xssSSkn7iswfGPmKilaN7j1tY80+DPxHvfAuneNfBHxU8bw+IL/wVYjxSNWilSVrnTJonklwFY5MM8dxHjIO3yv71cl4x8b/F/wAVePPgrf6l8PNO8O6Zq+uz3+lr/bDTSqp0u6ZIbyNYlEZKsHwjSBSCCSRmvTNd/ZW+GVxrthrHhHQdM8JxGzvtJ1uz0uwjhi1fTruExyW8oQrhg2x1cZKlTwcjGdoX7Onjmy8Q+CNX8S/G7UNfs/AE7NpNlNo0EQeFrd7cid0bdLL5b483AxgkLkkl+6TqaHg/9oceLNN+H9vb+GU/4SHxfc39vqWmrc4/sh7BXW+LsR83lzrHCAdu5pUOcVzuiftWyS+JNX8K67pfhOe8s9A1PXbdPD3iZNT8hrFVaS1u9sSeTIQ4wy7lyr+nPVaN+zR4S0P4g+O/iBa6nqPn+ObSS0NsJAqaaJlH2p7YjlXmkWORj6oK5Xw9+yNeaVNp7an8UZb630TwvqPhPTYIdCt7RYrW6iSPzXMbZkmXy0O44B54XJye6HvFzQP2kvEgvPDOo/Ev4c2/hjw74v8AD91rem30Ori8mhW2thdSrcRiNRHmHcw2s/3cHB4qxqPxY+Imr/C/XPG/ir4QwWPhC68K3+rwmHxLjURCLYyRRzIsQELyoTho5H8s9a6PVfgDouuWngLT9V1aeey8E6Td6OYDAAL6G4sBZyFucodmTxnk1haf8B/iNH4G1X4X638bZtV8MXPh+68P2MEuhW63MMUkPlRvNOpBlaNcDgR7urZPNLQfvFG2+NvjGHUf+EF+HHwgfXm0PwrpOutNd+IVgxDcRygQ+Y8bvJNiHCkjD/NuZMc3h+0bf+JrzRLL4T/Dm78U3OqeHrTxPepPqUOnpY2dySIIy7hg87lJQEGFHlklxkV1fhL4RQeFfFOp+KBrklydT8N6Z4dNu0G1USy87Eu7d1fzz8vbbwTXh/jL4ea38G73wrb+Bbj4krd2Pha28PXuteF/D9nqUGoxQO3lxS28zE28672aOX7uJCpJAAqkosHdHsvwi+L8fxk+H0vjWLw/d6CPt2o6ebO5kV542t7h4SXwNqsTGSVBO0nbljzXzn8JPjD430b9n/U/CPxQ1qY3uv8Ag7V9f8GeIfOYSXYSCV5rN5WORdwHDjnLxEMMFGx7Z+yt8NPEfw4+CemeGvFsM8GsT3moahcpcTLcSxtdXMkoSSRSA8gV1DsMAsCQal179lbwd4p/Z7s/2f8AxBqd5Pa6bZpb2WsIqx3drOu4pcJ2DDcwIHBVmUgqSDOibErtHLXP7UUfhO00XwtYxeGrzUdI8K6Zq2rSa/4nh0qW4eeEMtvaCVW86YhCxLFUXcilstx6X4N+Nem+OtaubLRNLlOnnwnpni2yupJNrTw3hnCxsoB2MvkDJyfvHj5a5TWP2ZbiLU7fWvA/j1NBvpNDstB1OS70O21KO7htI9kM6rN/qZlDMNykoRt3Idlbni34I+IdU8RWviXwd8UL7wzeSaDF4c1Z4tMt7j7daROzxuqkBYJlMkuHUFfnxtwBgtEd5djnvD37SPiTxXeeC7DQ/hvFJ/wk/hO28XX9zLqwjg0q0aYJKGJjzKwBym0DcQQdo+arXhn9oTxNqh8N+JtY+FsumeAfGV9BY6LrJ1RJbxTcHbZy3VqEHlJOxQLtkcqZE3Bc1v8Aw5+BWmfD6fQphrk+qf2J4Ph8HFJrdVSeKOXzPNYAnBboVrC0H9nTVtJufDuhan8VdS1PwJ4PvYb/AELw5JYwo8UkB3Wsc12DvnhgO0ou1T+6j3M23k0HZnK/DD4ya3ovgrwp4F8G+FJfF3i3V5/EN6tvPqAtLe0sLbVZ4jNPOyuyjc8UaqqOWJPQAkdNp/7TF54lg0fRPB/w5vb/AMaahc6naX+g3Wow2yaU2nSpDeNPcgOpRZJYVQojGTzVIAAbEUf7Mt1oVn4fu/BPxKvtA8S+Hjq0cOrrp8NxHc2moXjXctrNbyEqyLIUKkMCCmc8kHkPHnwpk+Flp4T1LwbL8Q77xFZ32p3d/wCJvD1pZ317LNflZLn7VaS4jmjmdE2hE/dGFMYGQXZE3Z13hT9p+71e9WDxP8O73w9EPHEnga5aa9ilayvDaxywPJsBVo5ZX8lWVsbniP8AHhfQ/hj8TE+JWnatrNrpn2TT7XXb7R9OnaYONQjtZDE9wuBwpkSUAc8IDnmvmhPhx400v9nfx54Yv/Bvi/UvEPxN8T3mpaOLmSKW+0+eQQC0vb6eEiGBo5YEuCUO1CAi5IxX0H4L+Fk/gu18EaR4c8WahaaH4W0ltPn0tYo/L1KQooE8zMpkDh1ZvlIyZHJzmiSighdnplFFFZmg3afWl2qe3vS0UAtBu0+tJT6Z+NABQBjgcUfjR+NAAAB0HvRgYBxyOh9KPxo/GiwCbR2HSk2AdM/nTvxo/GgBNg9aNo9M/Wl/GiiwBtUd6dhfak2+9G33oAjaGJiSyZJ6nvTvKG4t3JDE+9L+NPosA3Yc53HPrmkwvp696fTPxoAPLA7min0z8aACl+XGNvHpSfjR+NABgDtRgZBxwOg9KPxo/GiwC7T/AHjRsXn5RzTqM0AN2n1pwUjvRRQAzZ70bR6f5/yafTPxoAOMYzx70mOCD70v40Y96LAJsXqQCaXA49vejHvR+NFgDvnv60pQHqSaT8afQBCLWNZDKAd5GC245NPAA5A59afTPxoACAeCOOOKMf4UUUWAPrQQD157UfjR+NFgDHOaMD06Zx7UY4xmj8aLALtAzjj6UpXPXB/Om8+tPoATaKTafWnUUAM9qXjrj9aT8aPxosAeWM5zTGtIG6p2x+FTUUbARfZ0LFzySQc+4P8AhxT9pJzmnUUMBAADkCgop6gGlooFZ9xmADnAz0z7UvHp196T8aPxosMQAA5AOaRrVCCA23HTA/z6U7H+cU/vmgCubOM4JJIXoMdB6fSpEhC9WLdP0qSijcAooooA/9k="

# Full PNVR brand logo used as the page header/title
HEADER_LOGO_URI = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD/4SwARXhpZgAATU0AKgAAAAgABgALAAIAAAAmAAAIYgESAAMAAAABAAEAAAExAAIAAAAmAAAIiAEyAAIAAAAUAAAIrodpAAQAAAABAAAIwuocAAcAAAgMAAAAVgAAEUYc6gAAAAgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAFdpbmRvd3MgUGhvdG8gRWRpdG9yIDEwLjAuMTAwMTEuMTYzODQAV2luZG93cyBQaG90byBFZGl0b3IgMTAuMC4xMDAxMS4xNjM4NAAyMDI2OjA2OjEyIDE0OjIwOjMzAAAGkAMAAgAAABQAABEckAQAAgAAABQAABEwkpEAAgAAAAM3OAAAkpIAAgAAAAM3OAAAoAEAAwAAAAEAAQAA6hwABwAACAwAAAkQAAAAABzqAAAACAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAMjAyNjowNjoxMSAyMTozNzozMwAyMDI2OjA2OjExIDIxOjM3OjMzAAAAAAYBAwADAAAAAQAGAAABGgAFAAAAAQAAEZQBGwAFAAAAAQAAEZwBKAADAAAAAQAC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LTExVDIxOjM3OjMzLjc3ODwveG1wOkNyZWF0ZURhdGU+PC9yZGY6RGVzY3JpcHRpb24+PC9yZGY6UkRGPjwveDp4bXBtZXRhPg0KICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgPD94cGFja2V0IGVuZD0ndyc/Pv/bAEMAAwICAwICAwMDAwQDAwQFCAUFBAQFCgcHBggMCgwMCwoLCw0OEhANDhEOCwsQFhARExQVFRUMDxcYFhQYEhQVFP/bAEMBAwQEBQQFCQUFCRQNCw0UFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFP/AABEIAXAEAAMBIgACEQEDEQH/xAAfAAABBQEBAQEBAQAAAAAAAAAAAQIDBAUGBwgJCgv/xAC1EAACAQMDAgQDBQUEBAAAAX0BAgMABBEF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"

# ------------------- CUSTOM CSS (modernised) -------------------
st.markdown("""
<style>
/* Base */
html, body, .stApp {
    background: #f4f7fb !important;
    color: #1f2933 !important;
}
* { color: #1f2933; }
.block-container {
    padding-top: 1.4rem;
    padding-bottom: 3rem;
    max-width: 1540px;
}
[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #d9e2ec;
}
[data-testid="stSidebar"] * { color: #1f2933 !important; }
/* Top bar and header */
.hgdp-topbar {
    background: #0b4f6c;
    padding: 0.48rem 1rem;
    border-radius: 10px 10px 0 0;
    font-size: 0.88rem;
    letter-spacing: 0.01em;
}
.hgdp-topbar span { color: #ffffff !important; }
.hgdp-main-header {
    background: #ffffff;
    border: 1px solid #d9e2ec;
    border-top: 0;
    padding: 1.25rem 1.35rem 1.35rem 1.35rem;
    border-radius: 0 0 18px 18px;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.07);
    position: relative;
    overflow: hidden;
}
.hgdp-header-dna {
    position: absolute;
    top: -34px;
    right: -30px;
    width: 460px;
    height: 230px;
    opacity: 0.13;
    pointer-events: none;
    z-index: 0;
}
.hgdp-header-content {
    position: relative;
    z-index: 2;
}
.hgdp-kicker {
    color: #007C92 !important;
    font-size: 0.82rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.35rem;
}
.hgdp-main-title {
    margin: 0;
    font-size: 2.05rem;
    line-height: 1.12;
    color: #102a43 !important;
}
.hgdp-subtitle {
    margin-top: 0.58rem;
    max-width: 980px;
    color: #486581 !important;
    font-size: 1rem;
}
.hgdp-navstrip {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-top: 0.95rem;
}
.hgdp-navitem {
    background: #e6f7fb;
    color: #075985 !important;
    border: 1px solid #b6e3ef;
    border-radius: 999px;
    padding: 0.38rem 0.72rem;
    font-size: 0.86rem;
    font-weight: 700;
}
/* Logo-image header (replaces the text title) */
.hgdp-logo-header {
    background: #ffffff;
    border: 1px solid #d9e2ec;
    border-radius: 18px;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.07);
    padding: 1.1rem 1.35rem;
    text-align: center;
    margin-bottom: 0.4rem;
}
.hgdp-logo-img {
    width: 100%;
    max-width: 760px;
    height: auto;
    display: inline-block;
}
/* Alert banner */
.alert-banner {
    background: #fff3e0;
    border-left: 6px solid #f39c12;
    padding: 0.7rem 1rem;
    border-radius: 10px;
    margin: 1rem 0 1rem 0;
    font-weight: 500;
    box-shadow: 0 1px 2px rgba(0,0,0,0.05);
}
/* KPI cards */
.kpi-container {
    display: flex;
    gap: 1rem;
    margin: 1rem 0 1.5rem 0;
}
.kpi-card {
    background: white;
    border-radius: 18px;
    padding: 1rem 1.2rem;
    flex: 1;
    text-align: center;
    border: 1px solid #e2e8f0;
    box-shadow: 0 4px 12px rgba(0,0,0,0.03);
}
.kpi-value {
    font-size: 2rem;
    font-weight: 800;
    color: #0b4f6c;
}
.kpi-label {
    font-size: 0.85rem;
    font-weight: 600;
    color: #486581;
    margin-top: 0.25rem;
}
/* Search panel */
.hgdp-search-panel {
    margin-top: 1rem;
    margin-bottom: 0.7rem;
    padding: 1rem 1.1rem;
    border-radius: 16px;
    background: #ffffff;
    border: 1px solid #d9e2ec;
    box-shadow: 0 5px 18px rgba(15, 23, 42, 0.05);
}
.hgdp-section-label {
    color: #007C92 !important;
    font-weight: 800;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-bottom: 0.35rem;
}
.hgdp-badge {
    display: inline-block;
    padding: 0.22rem 0.58rem;
    border-radius: 999px;
    background: #e6f7fb;
    border: 1px solid #b6e3ef;
    color: #075985 !important;
    margin: 0.15rem 0.2rem 0.15rem 0;
    font-size: 0.85rem;
    font-weight: 700;
}
.hgdp-link-pill {
    display: inline-block;
    background: #ffffff;
    border: 1px solid #9fb3c8;
    color: #0b4f6c !important;
    border-radius: 999px;
    padding: 0.35rem 0.7rem;
    text-decoration: none !important;
    font-weight: 800;
    font-size: 0.84rem;
    margin-right: 0.5rem;
}
.hgdp-link-pill:hover {
    background: #e6f7fb;
    border-color: #007C92;
}
</style>
""", unsafe_allow_html=True)

# ------------------- DATA LOADING -------------------
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

DEFAULT_PARQUETS = [DATA_DIR / "neuro_mutations.parquet", PROJECT_DIR / "neuro_mutations.parquet"]
DEFAULT_CSVS = [DATA_DIR / "neuro_mutations.csv", DATA_DIR / "neuro_mutations csv.csv",
                PROJECT_DIR / "neuro_mutations.csv", PROJECT_DIR / "neuro_mutations csv.csv"]

CHR_LENGTHS = {
    "1": 248956422, "2": 242193529, "3": 198295559, "4": 190214555,
    "5": 181538259, "6": 170805979, "7": 159345973, "8": 145138636,
    "9": 138394717, "10": 133797422, "11": 135086622, "12": 133275309,
    "13": 114364328, "14": 107043718, "15": 101991189, "16": 90338345,
    "17": 83257441, "18": 80373285, "19": 58617616, "20": 64444167,
    "21": 46709983, "22": 50818468, "X": 156040895, "Y": 57227415,
}
CHR_ORDER = [str(i) for i in range(1, 23)] + ["X", "Y"]

SYSTEM_KEYWORDS = {
    "Brain / CNS": ["brain", "cns", "neurolog", "seizure", "epilep", "ataxia",
                    "developmental delay", "intellectual disability", "microcephaly",
                    "hypotonia", "spastic", "dystonia", "parkinson", "motor", "cerebell",
                    "autism", "encephal", "dementia"],
    "Peripheral nerve": ["peripheral neuropathy", "neuropathy", "charcot", "axon"],
    "Muscle": ["muscle", "myopathy", "dystrophy", "weakness", "myotonia"],
    "Eye": ["eye", "retina", "vision", "blind", "optic", "cataract", "nystagmus"],
    "Hearing / Ear": ["hearing", "deaf", "auditory", "ear"],
    "Heart": ["cardiac", "heart", "cardiomyopathy", "arrhythmia"],
    "Skeletal": ["bone", "skeletal", "spine", "craniofacial", "dysmorph"],
    "Metabolic / Liver": ["metabolic", "liver", "hepatic", "mitochondrial"],
}

def clean_chr(x):
    if pd.isna(x): return np.nan
    s = str(x).strip().upper().replace("CHR", "")
    m = re.match(r"^(X|Y|[0-9]{1,2})", s)
    if not m: return s
    val = m.group(1)
    return "X" if val == "23" else "Y" if val == "24" else val

def find_col(df, candidates):
    lower = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower: return lower[cand.lower()]
    for col in df.columns:
        for cand in candidates:
            if cand.lower() in col.lower(): return col
    return None

def safe_series(df, choices, fallback=""):
    for c in choices:
        if c in df.columns: return df[c].fillna(fallback).astype(str)
    return pd.Series([fallback] * len(df), index=df.index)

def infer_systems(df):
    disease = safe_series(df, ["Disease_filled", "Disease", "_disease"], "")
    phenotype = safe_series(df, ["PatientPhenotype", "Phenotype", "Clinical phenotype"], "")
    effect = safe_series(df, ["MutationEffect", "Effect", "Consequence"], "")
    text = (disease + " " + phenotype + " " + effect).str.lower()
    systems = []
    for item in text:
        hit = [sys for sys, keys in SYSTEM_KEYWORDS.items() if any(k in item for k in keys)]
        systems.append(", ".join(hit) if hit else "Unclassified")
    return systems

def add_display_fields(df):
    df = df.copy()
    df["_display_variant_class"] = safe_series(df, ["Variant_Class_Corrected", "Variant_Class", "MutationType", "MutationEffect", "_effect"], "")
    df["_display_ethnicity"] = safe_series(df, ["Ethnicity_Curated", "Ethnicity", "_ethnicity"], "")
    df["_display_reference"] = safe_series(df, ["Reference_Link_For_Streamlit", "Primary_Paper_URL", "PubMed_URL", "PMID_URL", "DOI_URL"], "")
    df["_display_paper_title"] = safe_series(df, ["Primary_Paper_Title", "Paper_Title", "Title"], "")
    df["_display_disease"] = safe_series(df, ["Disease_filled", "Disease", "_disease"], "")
    df["_display_omim"] = safe_series(df, ["OMIM_URL", "OMIM", "_omim"], "")
    df["_display_clinvar"] = safe_series(df, ["ClinVar_URL", "ClinVarID", "ClinVar", "_clinvar"], "")
    df["_display_pmid"] = safe_series(df, ["PMID", "_pmid"], "")
    # Consanguinity column
    if "Consanguinity" not in df.columns and "Consanguineous" not in df.columns:
        df["_consanguinity"] = "Unknown"
    else:
        cons_col = "Consanguinity" if "Consanguinity" in df.columns else "Consanguineous"
        df["_consanguinity"] = df[cons_col].fillna("Unknown").astype(str)
    # Evidence level
    if "Evidence_Level" not in df.columns:
        df["_evidence_level"] = df["_display_reference"].apply(lambda x: "Curated" if x else "Predicted")
    else:
        df["_evidence_level"] = df["Evidence_Level"]
    # Onset
    if "Onset_years" not in df.columns:
        df["_onset"] = np.nan
    else:
        df["_onset"] = pd.to_numeric(df["Onset_years"], errors="coerce")
    return df

@st.cache_data(show_spinner=False)
def load_csv(uploaded_file=None):
    source = uploaded_file
    if source is None:
        for p in DEFAULT_PARQUETS + DEFAULT_CSVS:
            if p.exists():
                source = p
                break
    if source is None:
        return None, None
    df = None
    if str(source).lower().endswith(".parquet"):
        df = pd.read_parquet(source)
    else:
        for enc in ["utf-8", "utf-8-sig", "latin1"]:
            try:
                df = pd.read_csv(source, encoding=enc)
                break
            except Exception:
                pass
    if df is None:
        return None, None
    df.columns = [str(c).strip() for c in df.columns]
    chr_col = find_col(df, ["Chr", "Chromosome", "chrom"])
    start_col = find_col(df, ["Start", "start", "Position", "pos"])
    end_col = find_col(df, ["End", "end"])
    gene_col = find_col(df, ["refGene", "Gene", "gene", "Gene Symbol", "symbol"])
    disease_col = find_col(df, ["Disease_filled", "Disease", "Phenotype", "Clinical phenotype"])
    effect_col = find_col(df, ["Variant_Class_Corrected", "MutationEffect", "Effect", "Consequence"])
    hgvs_col = find_col(df, ["HGVS", "Variant_Description", "Variant", "Mutation", "cDNA"])
    inheritance_col = find_col(df, ["Inheritance"])
    ethnicity_col = find_col(df, ["Ethnicity_Curated", "Ethnicity", "Population", "Region"])
    clinvar_col = find_col(df, ["ClinVarID", "ClinVar"])
    omim_col = find_col(df, ["OMIM"])
    pmid_col = find_col(df, ["PMID", "PubMed"])
    df["_chr"] = df[chr_col].apply(clean_chr) if chr_col else np.nan
    df["_start"] = pd.to_numeric(df[start_col], errors="coerce") if start_col else np.nan
    df["_end"] = pd.to_numeric(df[end_col], errors="coerce") if end_col else df["_start"]
    df["_gene"] = df[gene_col].astype(str).str.strip() if gene_col else "Unknown"
    df["_gene"] = df["_gene"].replace({"": "Unknown", "nan": "Unknown", "None": "Unknown"})
    df["_disease"] = df[disease_col].astype(str).str.strip() if disease_col else ""
    df["_effect"] = df[effect_col].astype(str).str.strip() if effect_col else ""
    df["_hgvs"] = df[hgvs_col].astype(str).str.strip() if hgvs_col else ""
    df["_inheritance"] = df[inheritance_col].astype(str).str.strip() if inheritance_col else ""
    df["_ethnicity"] = df[ethnicity_col].astype(str).str.strip() if ethnicity_col else ""
    df["_clinvar"] = df[clinvar_col].astype(str).str.strip() if clinvar_col else ""
    df["_omim"] = df[omim_col].astype(str).str.strip() if omim_col else ""
    df["_pmid"] = df[pmid_col].astype(str).str.strip() if pmid_col else ""
    df["_systems"] = infer_systems(df)
    df = add_display_fields(df)
    meta = {"chr_col": chr_col, "start_col": start_col, "end_col": end_col, "gene_col": gene_col,
            "disease_col": disease_col, "effect_col": effect_col, "hgvs_col": hgvs_col, "ethnicity_col": ethnicity_col}
    return df, meta

# ------------------- STATIC PUBLICATIONS LIST (data sources) -------------------
def get_static_publications():
    """Return a list of key publications from which the data has been sourced."""
    papers = [
        {
            "title": "A systematic review of hereditary neurological disorders diagnosed by whole exome sequencing in Pakistani population: updates from 2014 to November 2024",
            "link": "#",
            "date": "2024"
        },
        {
            "title": "Biallelic inheritance in a single Pakistani family with intellectual disability implicates new candidate gene RDH14",
            "link": "#",
            "date": "2024"
        },
        {
            "title": "Spectrum of neurological disorders in neurology outpatients clinics in urban and rural Sindh, Pakistan: a cross sectional study",
            "link": "#",
            "date": "2024"
        },
        {
            "title": "Exome sequencing reveals broad genetic heterogeneity for neuromuscular disorders in consanguineous Pakistani Families",
            "link": "#",
            "date": "2024"
        },
        {
            "title": "Clinical genomics expands the link between erroneous cell division, primary microcephaly and intellectual disability",
            "link": "#",
            "date": "2024"
        },
        {
            "title": "Biallelic variants identified in 36 Pakistani families and trios with autism spectrum disorder",
            "link": "#",
            "date": "2024"
        },
        {
            "title": "NGS driven molecular diagnosis of heterogeneous hereditary neurological disorders reveals novel and known variants in disease causing genes",
            "link": "#",
            "date": "2024"
        },
        {
            "title": "Exome sequencing of Pakistani consanguineous families identifies 30 novel candidate genes for recessive intellectual disability",
            "link": "#",
            "date": "2024"
        },
        {
            "title": "Evidence for autosomal recessive inheritance in SPG3A caused by homozygosity for a novel ATL1 missense mutation",
            "link": "#",
            "date": "2024"
        }
    ]
    return papers

# ------------------- ORIGINAL FUNCTIONS (unchanged) -------------------
def overview_panel(df):
    st.subheader("Database Overview")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Variants", f"{len(df):,}")
    c2.metric("Genes", f"{df['_gene'].replace('', np.nan).dropna().nunique():,}")
    c3.metric("Chromosomes", f"{df['_chr'].replace('', np.nan).dropna().nunique():,}")
    c4.metric("Diseases", f"{df['_display_disease'].replace('', np.nan).dropna().nunique():,}")
    c5.metric("Ethnicity Records", f"{df['_display_ethnicity'].replace('', np.nan).dropna().shape[0]:,}")
    ref_count = df["_display_reference"].replace("", np.nan).dropna().shape[0]
    st.progress(min(ref_count / max(len(df), 1), 1.0), text=f"Records with literature links: {ref_count:,} of {len(df):,}")

def chromosome_track_plot(df, chromosomes, key="chromosome_track_select", label_mode="Auto", max_labels_per_chromosome=12):
    selected_chromosomes = [c for c in CHR_ORDER if c in chromosomes]
    if not selected_chromosomes:
        st.warning("No chromosome selected.")
        return None
    plot_df = df.dropna(subset=["_chr", "_start"]).copy()
    plot_df = plot_df[plot_df["_chr"].isin(selected_chromosomes)].copy()
    plot_df["_mb"] = plot_df["_start"] / 1_000_000
    plot_df["_class"] = plot_df["_display_variant_class"].replace("", "Unknown").fillna("Unknown")
    compact_view = len(selected_chromosomes) > 8
    class_palette = {
        "Missense": "#2563eb", "Nonsense": "#dc2626", "Nonsense / stop-gain": "#dc2626",
        "Frameshift": "#7c2d12", "Frameshift deletion": "#7c2d12", "Splice-site / splicing": "#9333ea",
        "Splicing": "#9333ea", "Deletion": "#ea580c", "Insertion": "#16a34a", "Duplication": "#0891b2",
        "CNV / large region": "#475569", "Unknown": "#6b7280",
    }
    def color_for_class(x):
        x = str(x)
        for key_name, color in class_palette.items():
            if key_name.lower() in x.lower():
                return color
        return "#6b7280"
    def short_gene(x):
        x = str(x)
        return x if len(x) <= 15 else x[:14] + "…"
    def label_should_show():
        if label_mode == "Always":
            return True
        if label_mode == "Off":
            return False
        return len(selected_chromosomes) <= 8 and len(plot_df) <= 180
    show_labels = label_should_show()
    chrom_spacing = 2.4 if show_labels and not compact_view else 1.25
    x_positions = {chrom: i * chrom_spacing for i, chrom in enumerate(selected_chromosomes)}
    fig = go.Figure()
    for chrom in selected_chromosomes:
        x = x_positions[chrom]
        chrom_len = CHR_LENGTHS[chrom] / 1_000_000
        has_variants = chrom in set(plot_df["_chr"].astype(str))
        body_fill = "#f8fafc" if has_variants else "#f1f5f9"
        arm_lower = "#dbe4f0" if has_variants else "#f1f5f9"
        arm_upper = "#edf2fb" if has_variants else "#f8fafc"
        border = "#475569" if has_variants else "#cbd5e1"
        fig.add_shape(type="rect", x0=x-0.14, x1=x+0.14, y0=0, y1=chrom_len, line=dict(color=border, width=1.35), fillcolor=body_fill, layer="below")
        cap_h = max(chrom_len * 0.010, 1.0)
        fig.add_shape(type="circle", x0=x-0.14, x1=x+0.14, y0=-cap_h, y1=cap_h, line=dict(color=border, width=1.2), fillcolor=body_fill, layer="below")
        fig.add_shape(type="circle", x0=x-0.14, x1=x+0.14, y0=chrom_len-cap_h, y1=chrom_len+cap_h, line=dict(color=border, width=1.2), fillcolor=body_fill, layer="below")
        cent = chrom_len * 0.48
        gap = max(chrom_len * 0.016, 1.0)
        fig.add_shape(type="rect", x0=x-0.105, x1=x+0.105, y0=0, y1=max(cent-gap,0), line=dict(color="rgba(0,0,0,0)",width=0), fillcolor=arm_lower, layer="below")
        fig.add_shape(type="rect", x0=x-0.105, x1=x+0.105, y0=min(cent+gap, chrom_len), y1=chrom_len, line=dict(color="rgba(0,0,0,0)",width=0), fillcolor=arm_upper, layer="below")
        fig.add_shape(type="path", path=f"M {x},{cent - gap} L {x - 0.15},{cent} L {x},{cent + gap} L {x + 0.15},{cent} Z", line=dict(color="#475569", width=1.0), fillcolor="#cbd5e1", layer="below")
        fig.add_annotation(x=x, y=-max(16, chrom_len*0.075), text=f"<b>Chr {chrom}</b>", showarrow=False, font=dict(size=12, color="#111827"), xanchor="center", yanchor="top")
        if not has_variants:
            fig.add_annotation(x=x, y=chrom_len*0.55, text="no records", showarrow=False, textangle=-90, font=dict(size=10, color="#94a3b8"), xanchor="center", yanchor="middle")
    selected_gene = None
    if not plot_df.empty:
        plot_df["_rank"] = plot_df.groupby("_chr").cumcount()
        plot_df["_side"] = plot_df["_rank"] % 2
        for _, row in plot_df.iterrows():
            chrom = str(row["_chr"])
            x = x_positions[chrom]
            y = float(row["_mb"])
            color = color_for_class(row["_class"])
            fig.add_shape(type="line", x0=x-0.20, x1=x+0.20, y0=y, y1=y, line=dict(color=color, width=2.2), layer="above")
        for variant_class in sorted(plot_df["_class"].unique()):
            sub = plot_df[plot_df["_class"] == variant_class].copy()
            color = color_for_class(variant_class)
            xs, ys, custom = [], [], []
            for _, row in sub.iterrows():
                chrom = str(row["_chr"])
                base_x = x_positions[chrom]
                side = int(row["_side"])
                marker_x = base_x + 0.34 if side == 0 else base_x - 0.34
                xs.append(marker_x)
                ys.append(float(row["_mb"]))
                custom.append([str(row["_gene"]), str(row["_hgvs"]), str(row["_display_disease"]), str(row["_class"]), str(row["_display_ethnicity"]), str(row["_display_reference"]), str(row.name), str(row["_chr"])])
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers", marker=dict(size=8 if compact_view else 10, color=color, opacity=0.95, symbol="circle", line=dict(width=1.1, color="#ffffff")), name=str(variant_class)[:38], customdata=np.array(custom, dtype=object), hovertemplate="<b>Gene: %{customdata[0]}</b><br>Chromosome: %{customdata[7]}<br>Position: %{y:.2f} Mb<br>Variant: %{customdata[1]}<br>Class: %{customdata[3]}<br>Disease: %{customdata[2]}<br>Ethnicity: %{customdata[4]}<br>Reference: %{customdata[5]}<br><br><b>Click point for gene details</b><extra></extra>"))
        if show_labels:
            for chrom in selected_chromosomes:
                chrom_df = plot_df[plot_df["_chr"].astype(str) == chrom].copy()
                if chrom_df.empty:
                    continue
                chrom_len = CHR_LENGTHS[chrom] / 1_000_000
                base_x = x_positions[chrom]
                chrom_df = chrom_df.drop_duplicates(["_gene", "_start"]).copy()
                chrom_df = chrom_df.sort_values("_start")
                if len(chrom_df) > max_labels_per_chromosome:
                    chrom_df = chrom_df.head(max_labels_per_chromosome)
                left_rows = chrom_df.iloc[1::2].copy()
                right_rows = chrom_df.iloc[0::2].copy()
                for side_name, side_rows in [("left", left_rows), ("right", right_rows)]:
                    if side_rows.empty:
                        continue
                    side_rows = side_rows.sort_values("_start")
                    n = len(side_rows)
                    actual_y = side_rows["_mb"].astype(float).tolist()
                    min_y = max(4, min(actual_y) - 7)
                    max_y = min(chrom_len - 4, max(actual_y) + 7)
                    if n == 1:
                        label_y_positions = actual_y
                    else:
                        label_y_positions = np.linspace(min_y, max_y, n)
                    if side_name == "right":
                        label_x = base_x + 0.92
                        elbow_x = base_x + 0.48
                        xanchor = "left"
                        line_start_x = base_x + 0.20
                    else:
                        label_x = base_x - 0.92
                        elbow_x = base_x - 0.48
                        xanchor = "right"
                        line_start_x = base_x - 0.20
                    for (_, row), label_y in zip(side_rows.iterrows(), label_y_positions):
                        gene = short_gene(row["_gene"])
                        pos_y = float(row["_mb"])
                        color = color_for_class(row["_class"])
                        fig.add_shape(type="line", x0=line_start_x, x1=elbow_x, y0=pos_y, y1=label_y, line=dict(color="#334155", width=0.8), layer="above")
                        fig.add_shape(type="line", x0=elbow_x, x1=label_x, y0=label_y, y1=label_y, line=dict(color="#334155", width=0.8), layer="above")
                        fig.add_annotation(x=elbow_x, y=label_y, text=f"{pos_y:.1f}", showarrow=False, font=dict(size=8.5, color="#475569"), xanchor="right" if side_name == "right" else "left", yanchor="middle")
                        fig.add_annotation(x=label_x, y=label_y, text=gene, showarrow=False, font=dict(size=10, color=color), xanchor=xanchor, yanchor="middle")
    max_len = max(CHR_LENGTHS[c] for c in selected_chromosomes) / 1_000_000
    min_x = min(x_positions.values()) - (1.35 if show_labels else 0.75)
    max_x = max(x_positions.values()) + (1.35 if show_labels else 0.75)
    fig.update_layout(
        title="Publication-style vertical chromosome map with mutation labels",
        xaxis=dict(title="Chromosomes", tickmode="array", tickvals=[x_positions[c] for c in selected_chromosomes], ticktext=[f"Chr {c}" for c in selected_chromosomes], range=[min_x, max_x], showgrid=False),
        yaxis=dict(title="Genomic position, Mbp", range=[-max(22, max_len*0.09), max_len+max(16, max_len*0.05)], showgrid=True, gridcolor="#e5e7eb", zeroline=False),
        legend_title="Variant class", margin=dict(l=72, r=38, t=78, b=88), height=max(720, 34*len(selected_chromosomes)+360), clickmode="event+select", dragmode="select"
    )
    fig.update_layout(template="plotly_white", paper_bgcolor="rgba(255,255,255,0)", plot_bgcolor="#ffffff")
    event = st.plotly_chart(fig, width="stretch", key=key, on_select="rerun", selection_mode="points")
    try:
        points = event.get("selection", {}).get("points", [])
        if points:
            custom = points[0].get("customdata", [])
            if custom:
                selected_gene = custom[0]
    except Exception:
        selected_gene = None
    if selected_gene:
        st.session_state["selected_gene_from_track"] = selected_gene
    return selected_gene

def chromosome_tracks_page(df):
    st.subheader("Chromosome Track Explorer")
    present = [c for c in CHR_ORDER if c in set(df["_chr"].dropna().astype(str))]
    if not present:
        st.warning("No valid chromosomes found.")
        return
    c1, c2 = st.columns([1.35, 1])
    mode = c1.radio("Track mode", ["All chromosomes", "Single chromosome", "Selected chromosomes"], horizontal=True)
    show_empty = c2.checkbox("Show chromosomes without records", value=True)
    available = CHR_ORDER if show_empty else present
    if mode == "All chromosomes":
        selected = available
    elif mode == "Single chromosome":
        one = st.selectbox("Select chromosome", available)
        selected = [one]
    else:
        default = present[: min(6, len(present))]
        selected = st.multiselect("Select chromosomes", available, default=default)
    c3, c4 = st.columns([1,1])
    label_mode = c3.radio("Gene labels", ["Auto", "Always", "Off"], horizontal=True)
    max_labels = c4.slider("Max gene labels per chromosome", 3, 40, 12, 1)
    if not selected:
        st.info("Select at least one chromosome.")
        return
    selected_gene = chromosome_track_plot(df, selected, key="chromosome_tracks_page_select", label_mode=label_mode, max_labels_per_chromosome=max_labels)
    gene_from_state = selected_gene or st.session_state.get("selected_gene_from_track")
    if gene_from_state:
        st.write("")
        selected_gene_summary(df, gene_from_state)
    st.markdown("#### Records in selected chromosome(s)")
    view_cols = ["_gene", "_chr", "_start", "_end", "_hgvs", "_display_variant_class", "_display_disease", "_inheritance", "_display_ethnicity", "_display_reference"]
    sub = df[df["_chr"].isin(selected)].copy()
    if sub.empty:
        st.info("No mutation records found for the selected chromosome(s).")
    else:
        st.dataframe(sub[[c for c in view_cols if c in sub.columns]], width="stretch")

def genome_scatter(df):
    plot_df = df.dropna(subset=["_chr", "_start"]).copy()
    plot_df = plot_df[plot_df["_chr"].isin(CHR_LENGTHS.keys())]
    if plot_df.empty:
        st.warning("No valid chromosome/start coordinates found.")
        return
    plot_df["_chr"] = pd.Categorical(plot_df["_chr"], categories=CHR_ORDER, ordered=True)
    plot_df["_mb"] = plot_df["_start"] / 1_000_000
    fig = px.scatter(plot_df, x="_mb", y="_chr", color="_systems", hover_data={"_gene": True, "_hgvs": True, "_display_disease": True, "_display_variant_class": True, "_display_ethnicity": True, "_mb": ":.2f", "_chr": True}, labels={"_mb": "Genomic position, Mb", "_chr": "Chromosome", "_systems": "Affected system"}, title="Genome-wide mutation scatter", color_discrete_sequence=px.colors.qualitative.Bold)
    fig.update_traces(marker=dict(size=11, opacity=0.85, line=dict(width=1, color="#ffffff")))
    fig.update_layout(yaxis=dict(categoryorder="array", categoryarray=CHR_ORDER))
    fig.update_layout(
        template="plotly_white",
        title=dict(font=dict(size=18, color="#0b4f6c")),
        plot_bgcolor="#fbfdff",
        legend=dict(title="<b>Affected system</b>", bgcolor="rgba(255,255,255,0.85)", bordercolor="#cdd7e2", borderwidth=1),
        height=640, margin=dict(l=60, r=30, t=70, b=50),
        xaxis=dict(showgrid=True, gridcolor="#eef2f7"),
        yaxis_title="Chromosome"
    )
    st.plotly_chart(fig, width="stretch")

def selected_gene_summary(df, gene):
    if not gene:
        return
    gdf = df[df["_gene"].astype(str) == str(gene)].copy()
    if gdf.empty:
        return
    st.markdown(f"### Selected gene from chromosome track: `{gene}`")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Variants", len(gdf))
    c2.metric("Diseases", gdf["_display_disease"].replace("", np.nan).nunique())
    c3.metric("Chromosome(s)", gdf["_chr"].replace("", np.nan).nunique())
    c4.metric("Literature links", gdf["_display_reference"].replace("", np.nan).dropna().shape[0])
    organ_impact_panel(gdf)
    cols = ["_gene", "_chr", "_start", "_end", "_hgvs", "_display_variant_class", "_display_disease", "_inheritance", "_display_ethnicity", "_display_reference", "_display_paper_title"]
    st.dataframe(gdf[[c for c in cols if c in gdf.columns]], width="stretch")

def organ_impact_panel(gene_df):
    st.subheader("Human Organ / System Impact Panel")
    organ_order = ["Brain / CNS", "Peripheral nerve", "Muscle", "Eye", "Hearing / Ear", "Heart", "Skeletal", "Metabolic / Liver"]
    counts = {org: 0 for org in organ_order}
    system_text = " ".join(gene_df["_systems"].fillna("").astype(str).tolist()).lower()
    disease_text = " ".join(gene_df["_display_disease"].fillna("").astype(str).tolist()).lower()
    keyword_map = {
        "Brain / CNS": ["brain", "cns", "intellectual", "seizure", "epilep", "ataxia", "spastic", "microcephaly", "neuro", "development"],
        "Peripheral nerve": ["peripheral", "neuropathy", "charcot", "axon"],
        "Muscle": ["muscle", "myopathy", "dystrophy", "weakness"],
        "Eye": ["eye", "retina", "vision", "optic", "cataract"],
        "Hearing / Ear": ["hearing", "deaf", "auditory", "ear"],
        "Heart": ["heart", "cardiac", "cardiomyopathy"],
        "Skeletal": ["skeletal", "bone", "spine", "dysmorph"],
        "Metabolic / Liver": ["metabolic", "liver", "hepatic", "mitochondrial"],
    }
    for org in organ_order:
        if org.lower() in system_text:
            counts[org] += 2
        counts[org] += sum(1 for k in keyword_map[org] if k in disease_text)
    def level(score):
        if score >= 4: return "Strong"
        if score >= 2: return "Moderate"
        if score >= 1: return "Mild"
        return "No direct evidence"
    color_map = {"No direct evidence": "#f8fafc", "Mild": "#fee2e2", "Moderate": "#ef4444", "Strong": "#7f0019"}
    text_map = {"No direct evidence": "#111827", "Mild": "#7f1d1d", "Moderate": "#ffffff", "Strong": "#ffffff"}
    rows = [organ_order[:4], organ_order[4:]]
    for row in rows:
        cols = st.columns(4)
        for col, org in zip(cols, row):
            score = counts[org]
            lv = level(score)
            bg = color_map[lv]
            tc = text_map[lv]
            col.markdown(f"""
                <div style="background:{bg}; border:1px solid #e2e8f0; border-radius:18px; padding:1rem; min-height:110px; box-shadow:0 6px 16px rgba(15,23,42,0.06);">
                    <div style="font-weight:800; font-size:1rem; color:{tc} !important;">{org}</div>
                    <div style="margin-top:0.45rem; color:{tc} !important;">{lv}</div>
                    <div style="font-size:0.82rem; margin-top:0.25rem; color:{tc} !important;">Evidence score: {score}</div>
                </div>
            """, unsafe_allow_html=True)

def gene_card(df):
    genes = sorted([g for g in df["_gene"].dropna().unique() if g and g != "Unknown"])
    default_gene = st.session_state.get("selected_gene_from_track")
    default_index = genes.index(default_gene) if default_gene in genes else 0
    gene = st.selectbox("Select gene", genes, index=default_index)
    gdf = df[df["_gene"] == gene].copy()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Variants", len(gdf))
    c2.metric("Diseases", gdf["_display_disease"].replace("", np.nan).nunique())
    c3.metric("Chromosomes", gdf["_chr"].replace("", np.nan).nunique())
    c4.metric("Ethnicity records", gdf["_display_ethnicity"].replace("", np.nan).dropna().shape[0])
    st.markdown(f"### Gene Card: `{gene}`")
    badges = []
    for val in sorted(gdf["_display_variant_class"].replace("", np.nan).dropna().unique()):
        badges.append(f'<span class="hgdp-badge">{val}</span>')
    if badges:
        st.markdown(" ".join(badges), unsafe_allow_html=True)
    # external links (simplified)
    st.markdown('<div class="hgdp-section-label">External Resources</div>', unsafe_allow_html=True)
    gene_url = f"https://www.genecards.org/cgi-bin/carddisp.pl?gene={gene}"
    st.markdown(f'<a class="hgdp-link-pill" href="{gene_url}" target="_blank">GeneCards</a>', unsafe_allow_html=True)
    organ_impact_panel(gdf)
    st.markdown("#### Variant and Literature Records")
    view_cols = ["_chr", "_start", "_end", "_gene", "_hgvs", "_display_variant_class", "_effect", "_display_disease", "_inheritance", "_display_ethnicity", "_display_clinvar", "_display_omim", "_display_pmid", "_display_reference", "_display_paper_title", "_systems"]
    st.dataframe(gdf[[c for c in view_cols if c in gdf.columns]], width="stretch")

    if gene != "Unknown" and not gdf.empty and gdf["_chr"].notna().any():
        if st.button("🧬 View this gene in Genome Browser", key="jump_to_genome_browser"):
            st.session_state["genome_browser_locus"] = gene
            st.session_state["dashboard_section"] = "Genome Browser"
            st.rerun()

# ------------------- GENOME BROWSER (IGV.js) -------------------
IGV_JS_CDN = "https://cdn.jsdelivr.net/npm/igv@2.15.11/dist/igv.min.js"

def build_igv_features(df, max_features=8000):
    """Convert the PNVR mutation table into an IGV.js-compatible feature list (BED-style, 0-based start)."""
    work = df.dropna(subset=["_chr", "_start"]).copy()
    work = work[work["_chr"].isin(CHR_LENGTHS.keys())]
    if len(work) > max_features:
        work = work.head(max_features)
    feats = []
    for _, row in work.iterrows():
        try:
            start = int(row["_start"]) - 1  # BED is 0-based, half-open
            if start < 0:
                start = 0
            end_val = row["_end"] if pd.notna(row["_end"]) else row["_start"]
            end = int(end_val)
            if end <= start:
                end = start + 1
        except (ValueError, TypeError):
            continue
        feats.append({
            "chr": f"chr{row['_chr']}",
            "start": start,
            "end": end,
            "name": str(row.get("_gene", "") or "Unknown"),
            "hgvs": str(row.get("_hgvs", "") or ""),
            "variant_class": str(row.get("_display_variant_class", "") or ""),
            "disease": str(row.get("_display_disease", "") or ""),
            "ethnicity": str(row.get("_display_ethnicity", "") or ""),
            "inheritance": str(row.get("_inheritance", "") or ""),
        })
    return feats

def genome_browser_page(df, genome_build="hg38"):
    st.subheader("🧬 Genome Browser (IGV.js)")
    st.caption(
        "PNVR variants rendered as a live annotation track on the IGV.js genome browser "
        f"({genome_build.upper()}). Search a gene symbol or type a locus (e.g. `chr7:55,000,000-55,500,000`), "
        "then click a feature on the track for full variant details."
    )
    default_locus = st.session_state.pop("genome_browser_locus", "") if "genome_browser_locus" in st.session_state else ""
    if not default_locus:
        first_gene = df["_gene"].replace("", np.nan).dropna()
        default_locus = first_gene.iloc[0] if not first_gene.empty else "chr1"

    col1, col2 = st.columns([3, 1])
    with col1:
        locus = st.text_input("Locus / gene search", value=default_locus, key="igv_locus_input")
    with col2:
        max_feats = st.number_input("Max variants on track", min_value=100, max_value=20000, value=5000, step=500)

    feats = build_igv_features(df, max_features=int(max_feats))
    if not feats:
        st.warning("No variants with valid chromosome/coordinate data are available to plot in the genome browser.")
        return

    features_json = json.dumps(feats)
    safe_locus = json.dumps(locus.strip() if locus and locus.strip() else "chr1")

    html_code = f"""
    <div id="igv-div" style="width:100%; height:640px; border:1px solid #d7e3ea; border-radius:8px;"></div>
    <script src="{IGV_JS_CDN}"></script>
    <script>
      (function() {{
        var igvDiv = document.getElementById("igv-div");
        var options = {{
          genome: "{genome_build}",
          locus: {safe_locus},
          tracks: [
            {{
              name: "PNVR Variants",
              type: "annotation",
              displayMode: "EXPANDED",
              color: "#0f6286",
              altColor: "#ef9c1f",
              height: 260,
              features: {features_json}
            }}
          ]
        }};
        igv.createBrowser(igvDiv, options).then(function (browser) {{
          window._pnvrIgvBrowser = browser;
        }});
      }})();
    </script>
    """
    components.html(html_code, height=680, scrolling=True)

    st.markdown(
        "**Notes:** the browser loads the public IGV.js library and the "
        f"{genome_build.upper()} reference sequence/gene annotations from the IGV.js CDN, so this page "
        "requires the *end user's* browser to have internet access (not this server). "
        "The PNVR variant track itself is generated live from your currently filtered dataset above."
    )
    with st.expander("🔌 Embed / share this genome browser view", expanded=False):
        st.markdown(
            "- To expose this same track to **external** genome browsers (e.g. a colleague's local IGV desktop "
            "app, or JBrowse2), export the filtered table below as a BED file and host it at a stable URL, "
            "then load that URL as a custom track.\n"
            "- For a REST-style API other tools can query directly, see the **Data Resources** section."
        )
        bed_lines = ["\t".join([f["chr"], str(f["start"]), str(f["end"]), f["name"].replace(" ", "_")]) for f in feats]
        bed_text = "\n".join(bed_lines)
        st.download_button(
            "⬇️ Download current view as BED", data=bed_text, file_name="pnvr_variants.bed", mime="text/plain"
        )

def phenotype_explorer(df):
    st.subheader("Disease and Phenotype Overview")
    c1, c2, c3 = st.columns([1, 1.2, 1.2])
    top_n = c1.slider("Top diseases", 10, 50, 25, 5)
    search = c2.text_input("Search disease, gene, variant, or PMID", "")
    show_table = c3.checkbox("Show detailed mutation table", value=True)
    work = df.copy()
    if search.strip():
        q = search.strip().lower()
        work = work[work["_display_disease"].fillna("").str.lower().str.contains(q, regex=False) | work["_gene"].fillna("").str.lower().str.contains(q, regex=False) | work["_hgvs"].fillna("").str.lower().str.contains(q, regex=False) | work["_display_pmid"].fillna("").str.lower().str.contains(q, regex=False)]
    disease_counts = work["_display_disease"].replace("", np.nan).dropna().value_counts().head(top_n).reset_index()
    disease_counts.columns = ["Disease / phenotype", "Mutation count"]
    if not disease_counts.empty:
        disease_counts["Short label"] = disease_counts["Disease / phenotype"].apply(lambda x: x if len(str(x)) <= 62 else str(x)[:59]+"...")
        fig = px.bar(disease_counts, x="Mutation count", y="Short label", orientation="h",
                     hover_data=["Disease / phenotype"], title="Top diseases and phenotypes",
                     color="Mutation count", color_continuous_scale="Sunset")
        fig.update_traces(marker_line_color="#ffffff", marker_line_width=0.6)
        fig.update_layout(height=max(430, 27*len(disease_counts)), yaxis=dict(autorange="reversed"),
                          margin=dict(l=20, r=30, t=60, b=40),
                          title=dict(font=dict(size=18, color="#0b4f6c")),
                          plot_bgcolor="#fbfdff", coloraxis_showscale=False,
                          xaxis=dict(showgrid=True, gridcolor="#eef2f7"))
        st.plotly_chart(fig, width="stretch")
    sys_counts = work["_systems"].replace("", np.nan).dropna().value_counts().reset_index()
    sys_counts.columns = ["Affected system", "Mutation count"]
    fig2 = px.bar(sys_counts, x="Affected system", y="Mutation count", text="Mutation count",
                  title="Affected organ/system distribution",
                  color="Affected system", color_discrete_sequence=px.colors.qualitative.Vivid)
    fig2.update_traces(marker_line_color="#ffffff", marker_line_width=0.8, textposition="outside")
    fig2.update_layout(height=460, xaxis_tickangle=-25, showlegend=False,
                       title=dict(font=dict(size=18, color="#0b4f6c")),
                       plot_bgcolor="#fbfdff", margin=dict(l=40, r=30, t=60, b=90),
                       yaxis=dict(showgrid=True, gridcolor="#eef2f7"))
    st.plotly_chart(fig2, width="stretch")
    if show_table:
        cols = ["_gene", "_chr", "_start", "_hgvs", "_display_variant_class", "_display_disease", "_inheritance", "_display_ethnicity", "_systems", "_display_reference"]
        st.dataframe(work[[c for c in cols if c in work.columns]], width="stretch")

def quality_page(df, meta):
    st.subheader("Data Quality")
    checks = {
        "Rows": int(len(df)), "Columns": int(len(df.columns)), "Missing chromosome": int(df["_chr"].isna().sum()),
        "Missing start coordinate": int(df["_start"].isna().sum()), "Unknown gene": int((df["_gene"] == "Unknown").sum()),
        "Missing disease": int((df["_display_disease"].fillna("") == "").sum()),
        "Missing ethnicity": int((df["_display_ethnicity"].fillna("") == "").sum()),
        "Missing literature link": int((df["_display_reference"].fillna("") == "").sum()),
    }
    st.json(checks)
    st.subheader("Detected source columns")
    st.json(meta)
    bad_chr = df[~df["_chr"].isin(CHR_LENGTHS.keys())]
    if not bad_chr.empty:
        st.warning("Rows with non-standard chromosome labels")
        st.dataframe(bad_chr.head(100), width="stretch")

# ------------------- ENHANCED CIRCOS PLOT (classic style with multiple rings) -------------------
def circos_multiring_plot(df, max_points_total=3000):
    """
    Create a publication‑ready Circos plot with:
    - Chromosome arcs with p/q arms and centromere gap.
    - Separate concentric rings for each variant class.
    - Chromosome position ticks and labels.
    """
    # Define target classes and mapping
    target_classes = [
        "copy number variation",
        "frameshift deletion",
        "frameshift duplication",
        "missense",
        "nonsense",
        "SNV",
        "splice site/splicing"
    ]
    
    def map_variant_class(v):
        if pd.isna(v):
            return None
        v_low = v.lower()
        if "cnv" in v_low or "copy number" in v_low:
            return "copy number variation"
        if "frameshift deletion" in v_low:
            return "frameshift deletion"
        if "frameshift duplication" in v_low:
            return "frameshift duplication"
        if "missense" in v_low:
            return "missense"
        if "nonsense" in v_low or "stop-gain" in v_low:
            return "nonsense"
        if v_low == "snv" or "single nucleotide" in v_low:
            return "SNV"
        if "splice" in v_low:
            return "splice site/splicing"
        return None
    
    # Prepare data
    plot_df = df.dropna(subset=["_chr", "_start"]).copy()
    plot_df = plot_df[plot_df["_chr"].isin(CHR_LENGTHS.keys())]
    if plot_df.empty:
        st.warning("No data for Circos plot.")
        return None
    
    plot_df["_ring_class"] = plot_df["_display_variant_class"].apply(map_variant_class)
    plot_df = plot_df[plot_df["_ring_class"].notna()]
    if plot_df.empty:
        st.warning("No variants from the requested classes (CNV, frameshift deletions/duplications, missense, nonsense, SNV, splicing).")
        return None
    
    if len(plot_df) > max_points_total:
        plot_df = plot_df.sample(n=max_points_total, random_state=42)
        st.info(f"Showing {len(plot_df)} points (limited for performance).")
    
    # Chromosome order and angles
    chromosomes = [c for c in CHR_ORDER if c in CHR_LENGTHS]
    n_chr = len(chromosomes)
    angle_per_chr = 360.0 / n_chr
    start_angles = {chr: i * angle_per_chr for i, chr in enumerate(chromosomes)}
    end_angles = {chr: (i+1) * angle_per_chr for i, chr in enumerate(chromosomes)}
    
    # Normalise positions 0..1
    plot_df["_norm_pos"] = plot_df.apply(lambda row: row["_start"] / CHR_LENGTHS[row["_chr"]], axis=1)
    
    def variant_angle(row):
        start_ang = start_angles[row["_chr"]]
        end_ang = end_angles[row["_chr"]]
        return start_ang + row["_norm_pos"] * (end_ang - start_ang)
    
    plot_df["_angle_deg"] = plot_df.apply(variant_angle, axis=1)
    plot_df["_angle_rad"] = plot_df["_angle_deg"] * math.pi / 180.0
    
    # Vivid, high-contrast colors for the data rings
    class_colors = {
        "copy number variation": "#A0522D",
        "frameshift deletion": "#FF7043",
        "frameshift duplication": "#FFB300",
        "missense": "#2196F3",
        "nonsense": "#E53935",
        "SNV": "#43A047",
        "splice site/splicing": "#AB47BC"
    }
    # Soft "glow" companion colours (lighter tints used behind the markers)
    class_glow = {
        "copy number variation": "#D7A98C",
        "frameshift deletion": "#FFB59B",
        "frameshift duplication": "#FFE08A",
        "missense": "#90CAF9",
        "nonsense": "#F4A6A4",
        "SNV": "#A5D6A7",
        "splice site/splicing": "#E1BEE7"
    }
    present_classes = [c for c in target_classes if c in plot_df["_ring_class"].unique()]
    n_rings = len(present_classes)
    if n_rings == 0:
        st.warning("None of the requested variant classes are present.")
        return None
    
    # Ring radii: from 0.35 to 0.80 (inside the chromosome arcs)
    ring_radii = np.linspace(0.35, 0.80, n_rings)
    radius_dict = {cls: ring_radii[i] for i, cls in enumerate(present_classes)}
    
    # Chromosome arc parameters
    radius_outer = 1.05
    radius_inner = 0.85
    centromere_gap_frac = 0.10   # 10% of the chromosome arc is the centromere gap

    # Classic colourful ideogram: every chromosome gets its own hue.
    # The p (short) arm uses a lighter tint, the q (long) arm a deeper shade
    # of the same hue, which reads as a realistic, 3D-shaded chromosome band.
    import colorsys
    def _hue_pair(i, n):
        h = (i / max(n, 1)) % 1.0
        # lighter p-arm
        r1, g1, b1 = colorsys.hls_to_rgb(h, 0.74, 0.85)
        # deeper q-arm
        r2, g2, b2 = colorsys.hls_to_rgb(h, 0.52, 0.95)
        to_hex = lambda r, g, b: "#%02x%02x%02x" % (int(r*255), int(g*255), int(b*255))
        return to_hex(r1, g1, b1), to_hex(r2, g2, b2)
    chr_arm_colors = {c: _hue_pair(i, n_chr) for i, c in enumerate(chromosomes)}
    
    fig = go.Figure()
    
    # ---- Draw chromosome arcs (p and q arms) ----
    for chr_name in chromosomes:
        start_ang = start_angles[chr_name]
        end_ang = end_angles[chr_name]
        span = end_ang - start_ang
        p_arm_color, q_arm_color = chr_arm_colors[chr_name]
        # Split the arc into p (40%) and q (60%) with a small gap
        p_fraction = 0.40
        q_fraction = 0.60
        gap_deg = span * centromere_gap_frac
        p_end_ang = start_ang + span * p_fraction
        q_start_ang = p_end_ang + gap_deg
        # Adjust q_end to keep total span correct (include the gap)
        q_end_ang = start_ang + span + gap_deg
        # Draw p arm
        theta_p = np.linspace(start_ang, p_end_ang, 40) * math.pi/180
        x_outer_p = radius_outer * np.cos(theta_p)
        y_outer_p = radius_outer * np.sin(theta_p)
        x_inner_p = radius_inner * np.cos(theta_p)
        y_inner_p = radius_inner * np.sin(theta_p)
        fig.add_trace(go.Scatter(
            x=np.concatenate([x_outer_p, x_inner_p[::-1]]),
            y=np.concatenate([y_outer_p, y_inner_p[::-1]]),
            fill="toself",
            fillcolor=p_arm_color,
            line=dict(color="#ffffff", width=1.1),
            name=f"Chr {chr_name} p",
            showlegend=False,
            hoverinfo="text",
            text=f"Chromosome {chr_name} short arm"
        ))
        # Draw q arm
        theta_q = np.linspace(q_start_ang, q_end_ang, 40) * math.pi/180
        x_outer_q = radius_outer * np.cos(theta_q)
        y_outer_q = radius_outer * np.sin(theta_q)
        x_inner_q = radius_inner * np.cos(theta_q)
        y_inner_q = radius_inner * np.sin(theta_q)
        fig.add_trace(go.Scatter(
            x=np.concatenate([x_outer_q, x_inner_q[::-1]]),
            y=np.concatenate([y_outer_q, y_inner_q[::-1]]),
            fill="toself",
            fillcolor=q_arm_color,
            line=dict(color="#ffffff", width=1.1),
            name=f"Chr {chr_name} q",
            showlegend=False,
            hoverinfo="text",
            text=f"Chromosome {chr_name} long arm"
        ))
        # Centromere marker (a small crimson constriction in the gap)
        cen_ang = (p_end_ang + q_start_ang) / 2 * math.pi / 180
        cen_r = (radius_outer + radius_inner) / 2
        fig.add_trace(go.Scatter(
            x=[cen_r * np.cos(cen_ang)], y=[cen_r * np.sin(cen_ang)],
            mode="markers",
            marker=dict(size=7, color="#c62828", symbol="diamond",
                        line=dict(color="#ffffff", width=1)),
            showlegend=False, hoverinfo="text",
            text=f"Chromosome {chr_name} centromere"
        ))
        # Chromosome label at outer edge
        mid_ang = (start_ang + end_ang) / 2 * math.pi/180
        label_radius = radius_outer + 0.08
        x_label = label_radius * np.cos(mid_ang)
        y_label = label_radius * np.sin(mid_ang)
        fig.add_annotation(
            x=x_label, y=y_label, text=f"<b>{chr_name}</b>", showarrow=False,
            font=dict(size=12.5, color="#1a2733"), xanchor="center", yanchor="middle"
        )
    
    # ---- Add tick marks at 10 Mb intervals along each chromosome ----
    for chr_name in chromosomes:
        start_ang = start_angles[chr_name]
        end_ang = end_angles[chr_name]
        length = CHR_LENGTHS[chr_name]
        step = 10_000_000
        for pos in range(step, length, step):
            frac = pos / length
            ang_deg = start_ang + frac * (end_ang - start_ang)
            ang_rad = ang_deg * math.pi/180
            x0 = radius_outer * np.cos(ang_rad)
            y0 = radius_outer * np.sin(ang_rad)
            x1 = (radius_outer + 0.05) * np.cos(ang_rad)
            y1 = (radius_outer + 0.05) * np.sin(ang_rad)
            fig.add_shape(type="line", x0=x0, y0=y0, x1=x1, y1=y1,
                          line=dict(color="#888", width=1))
            # Add label every 50 Mb
            if pos % 50_000_000 == 0:
                label_radius = radius_outer + 0.12
                xl = label_radius * np.cos(ang_rad)
                yl = label_radius * np.sin(ang_rad)
                fig.add_annotation(x=xl, y=yl, text=f"{pos/1e6:.0f} Mb",
                                   showarrow=False, font=dict(size=8, color="#555"),
                                   xanchor="center", yanchor="middle")
    
    # ---- Add data rings for each variant class ----
    for class_name in present_classes:
        class_df = plot_df[plot_df["_ring_class"] == class_name].copy()
        ring_rad = radius_dict[class_name]
        xs = ring_rad * np.cos(class_df["_angle_rad"])
        ys = ring_rad * np.sin(class_df["_angle_rad"])
        color = class_colors[class_name]
        glow = class_glow.get(class_name, color)
        # Soft glow halo behind the points
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers",
            marker=dict(size=9, color=glow, opacity=0.35,
                        line=dict(width=0)),
            showlegend=False, hoverinfo="skip"
        ))
        # Crisp coloured points on top
        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="markers",
            marker=dict(size=5, color=color, opacity=0.92,
                        line=dict(color="#ffffff", width=0.6)),
            name=class_name,
            text=class_df["_gene"],
            hoverinfo="text",
            hovertext=class_df.apply(
                lambda row: f"<b>Class:</b> {class_name}<br>"
                            f"<b>Gene:</b> {row['_gene']}<br>"
                            f"<b>Chromosome:</b> {row['_chr']}<br>"
                            f"<b>Position:</b> {row['_start']:,}<br>"
                            f"<b>Disease:</b> {row['_display_disease']}<br>"
                            f"<b>Variant:</b> {row['_hgvs']}<br>"
                            f"<b>Ethnicity:</b> {row['_display_ethnicity']}",
                axis=1
            )
        ))
    
    # ---- Add circular grid lines (subtle, dashed) ----
    for r in np.arange(0.2, 1.1, 0.2):
        theta = np.linspace(0, 2*math.pi, 120)
        x_grid = r * np.cos(theta)
        y_grid = r * np.sin(theta)
        fig.add_trace(go.Scatter(
            x=x_grid, y=y_grid, mode="lines",
            line=dict(color="#cdd7e2", width=0.6, dash="dot"),
            showlegend=False, hoverinfo="none"
        ))
    
    # Final layout
    fig.update_layout(
        title=dict(
            text="Classic Circos Plot — Neurogenetic Variants by Chromosome and Mutation Class",
            font=dict(size=17, color="#0b4f6c"), x=0.5, xanchor="center"
        ),
        xaxis=dict(visible=False, range=[-1.25, 1.25], scaleanchor="y", scaleratio=1),
        yaxis=dict(visible=False, range=[-1.25, 1.25]),
        showlegend=True,
        legend=dict(
            title=dict(text="<b>Variant class</b>", font=dict(color="#0b4f6c")),
            bgcolor="rgba(255,255,255,0.85)", bordercolor="#cdd7e2", borderwidth=1,
            font=dict(size=11)
        ),
        height=760,
        width=760,
        margin=dict(l=30, r=30, t=70, b=30),
        plot_bgcolor="#fbfdff",
        paper_bgcolor="#ffffff"
    )
    return fig

# ------------------- NEW ENHANCEMENTS (HPO, Map, Submit, etc.) -------------------
def render_hpo_tree():
    st.markdown("**🧠 Neurological Disease Tree (HPO)**")
    hpo_categories = {
        "Seizures / Epilepsy": ["Generalized seizure", "Focal seizure", "Infantile spasm"],
        "Movement disorders": ["Ataxia", "Dystonia", "Parkinsonism", "Chorea"],
        "Neurodevelopmental": ["Intellectual disability", "Developmental delay", "Microcephaly", "Macrocephaly"],
        "Peripheral neuropathy": ["Charcot-Marie-Tooth", "Hereditary sensory neuropathy"],
        "Muscle disorders": ["Muscular dystrophy", "Myopathy", "Congenital hypotonia"]
    }
    selected = []
    for cat, terms in hpo_categories.items():
        with st.expander(cat):
            for term in terms:
                if st.checkbox(term, key=f"hpo_{term}"):
                    selected.append(term)
    return selected

def render_region_map(df):
    st.markdown("**🗺️ Pakistani Region Map**")
    if "_ethnicity" in df.columns:
        region_counts = df["_ethnicity"].dropna().value_counts().reset_index()
        region_counts.columns = ["Region", "Variant count"]
        coords = {
            "Punjab": (31.1471, 72.6867), "Sindh": (25.8943, 68.5247), "KPK": (34.1355, 72.5225),
            "Balochistan": (28.494, 65.414), "Gilgit-Baltistan": (35.9197, 74.375), "Islamabad": (33.6844, 73.0479),
            "Pashtun": (34.1355, 71.5247), "Urdu speaking": (24.8607, 67.0011), "Kashmir": (34.2479, 73.7277)
        }
        region_counts["lat"] = region_counts["Region"].map(lambda x: coords.get(x, (30.3753, 69.3451))[0])
        region_counts["lon"] = region_counts["Region"].map(lambda x: coords.get(x, (30.3753, 69.3451))[1])
        fig = px.scatter_mapbox(region_counts, lat="lat", lon="lon", size="Variant count",
                                 color="Variant count", color_continuous_scale="Turbo",
                                 size_max=34, hover_name="Region", zoom=4.6, height=320,
                                 title="Variants by Pakistani region")
        fig.update_layout(mapbox_style="carto-positron", margin=dict(l=0, r=0, t=40, b=0),
                          coloraxis_showscale=False,
                          title=dict(font=dict(size=15, color="#0b4f6c")))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Ethnicity/region data not available for map.")

def submit_new_variant_form():
    with st.expander("🧬 Submit a New Variant (for curation)", expanded=False):
        with st.form("submit_variant"):
            gene = st.text_input("Gene symbol")
            variant = st.text_input("Variant (HGVS)")
            disease = st.text_input("Disease")
            onset = st.number_input("Age of onset (years)", min_value=0, step=1)
            region = st.text_input("Pakistani region/ethnicity")
            consang = st.selectbox("Consanguinity", ["Yes", "No", "Unknown"])
            inheritance = st.selectbox("Inheritance", ["AR", "AD", "XL", "Mitochondrial", "Unknown"])
            evidence = st.selectbox("Evidence level", ["Gold (peer-reviewed)", "Silver (single family)", "Predicted"])
            submitted = st.form_submit_button("Submit for review")
            if submitted:
                new_record = {
                    "Gene": gene, "Variant": variant, "Disease": disease, "Onset": onset,
                    "Region": region, "Consanguinity": consang, "Inheritance": inheritance,
                    "Evidence_Level": evidence, "Submission_date": datetime.now().isoformat()
                }
                submissions_file = DATA_DIR / "user_submissions.csv"
                if submissions_file.exists():
                    df_sub = pd.read_csv(submissions_file)
                    df_sub = pd.concat([df_sub, pd.DataFrame([new_record])], ignore_index=True)
                else:
                    df_sub = pd.DataFrame([new_record])
                df_sub.to_csv(submissions_file, index=False)
                st.success("Variant submitted. Thank you!")

# ------------------- DATA RESOURCES CATALOG (directory of databases) -------------------
def get_data_resources():
    """Curated directory of genetics & genomics databases relevant to
    Pakistani / South Asian neurogenetics. Descriptions are factual and
    each entry links to the official resource."""
    return [
        # ---- This repository ----
        {"cat": "Pakistan-focused resources", "name": "PNVR", "acr": "this atlas",
         "full": "Pakistan Neurogenomic Variant Repository",
         "desc": "This resource. An evidence-linked atlas of neurogenetic variants curated from the Pakistani population, with chromosome tracks, phenotype mapping and publication links.",
         "contents": "Curated neurogenetic variants, genes, diseases, inheritance, Pakistani population context, literature evidence.",
         "manager": "Curated locally (see Data Sources – Key Publications on the Home page).",
         "region": "Pakistan", "access": "Open", "url": "#"},
        {"cat": "Pakistan-focused resources", "name": "PGMD", "acr": "PGMD",
         "full": "Pakistan Genetic Mutation Database",
         "desc": "A centralized Pakistani 'mutome' compiling disease-causing mutations reported in Pakistani families, developed to address the country's high rate of consanguinity and autosomal-recessive disease burden.",
         "contents": "Published mutations from Pakistani ethnic families across many Mendelian disorders.",
         "manager": "Pakistani research groups (described in Meta Gene, 2017).",
         "region": "Pakistan", "access": "Open",
         "url": "https://www.sciencedirect.com/science/article/abs/pii/S1769721217303312"},

        # ---- Core clinical variant & disease databases ----
        {"cat": "Core clinical variant & disease databases", "name": "ClinVar", "acr": "ClinVar",
         "full": "ClinVar — variant–phenotype clinical archive",
         "desc": "Public archive of relationships between human genetic variants and phenotypes, with supporting evidence and clinical-significance classifications aggregated from submitters worldwide.",
         "contents": "Variant–condition assertions, pathogenicity classifications, review status, supporting evidence.",
         "manager": "NCBI, National Institutes of Health (USA).",
         "region": "International", "access": "Open", "url": "https://www.ncbi.nlm.nih.gov/clinvar/"},
        {"cat": "Core clinical variant & disease databases", "name": "OMIM", "acr": "OMIM",
         "full": "Online Mendelian Inheritance in Man",
         "desc": "Comprehensive, continuously updated catalog of human genes and genetic phenotypes, focusing on the relationship between genes and Mendelian disorders.",
         "contents": "Gene and phenotype entries, gene–disease relationships, allelic variants, inheritance.",
         "manager": "Johns Hopkins University (McKusick-Nathans Institute).",
         "region": "International", "access": "Free (academic)", "url": "https://www.omim.org/"},
        {"cat": "Core clinical variant & disease databases", "name": "ClinGen", "acr": "ClinGen",
         "full": "Clinical Genome Resource",
         "desc": "NIH-funded expert-curation effort that defines the clinical relevance of genes and variants — gene–disease validity, variant pathogenicity and dosage sensitivity — for use in precision medicine.",
         "contents": "Gene–disease validity classifications, variant curations, dosage-sensitivity calls, actionability.",
         "manager": "NHGRI / NIH-supported international consortium.",
         "region": "International", "access": "Open", "url": "https://clinicalgenome.org/"},
        {"cat": "Core clinical variant & disease databases", "name": "DECIPHER", "acr": "DECIPHER",
         "full": "Database of genomiC varIation and Phenotype in Humans using Ensembl Resources",
         "desc": "Interactive web platform for sharing and interpreting genomic variants and phenotypes in patients with rare disorders, linking variants to standardized (HPO) phenotypes.",
         "contents": "CNVs, sequence variants, patient phenotypes, genome-browser context.",
         "manager": "Wellcome Sanger Institute / EMBL-EBI (UK).",
         "region": "International", "access": "Open", "url": "https://www.deciphergenomics.org/"},
        {"cat": "Core clinical variant & disease databases", "name": "LOVD", "acr": "LOVD",
         "full": "Leiden Open Variation Database",
         "desc": "Open-source platform that hosts many gene-centered variant databases, allowing the community to collect and display DNA variations and their clinical effects.",
         "contents": "Gene-specific variant records, observed phenotypes, frequencies, curators' notes.",
         "manager": "Leiden University Medical Center (Netherlands).",
         "region": "International", "access": "Open", "url": "https://www.lovd.nl/"},
        {"cat": "Core clinical variant & disease databases", "name": "HGMD", "acr": "HGMD",
         "full": "Human Gene Mutation Database",
         "desc": "Catalog of published germline mutations responsible for human inherited disease. A free public version is available; the comprehensive Professional version is licensed.",
         "contents": "Disease-causing and disease-associated germline variants curated from the literature.",
         "manager": "Cardiff University (UK); Professional version via QIAGEN.",
         "region": "International", "access": "Public + Licensed", "url": "http://www.hgmd.cf.ac.uk/"},
        {"cat": "Core clinical variant & disease databases", "name": "Orphanet", "acr": "Orphanet",
         "full": "Orphanet — rare diseases & orphan drugs portal",
         "desc": "Reference portal for information on rare diseases and orphan drugs, providing a standard nomenclature (ORPHA codes), gene–disease associations and epidemiology.",
         "contents": "Rare-disease nomenclature, gene–disease links, epidemiology, expert centres.",
         "manager": "INSERM (France) and the Orphanet consortium.",
         "region": "International", "access": "Open", "url": "https://www.orpha.net/"},
        {"cat": "Core clinical variant & disease databases", "name": "GeneReviews", "acr": "GeneReviews",
         "full": "GeneReviews — clinical summaries of inherited conditions",
         "desc": "Expert-authored, peer-reviewed clinical descriptions of inherited conditions, covering diagnosis, management and genetic counseling.",
         "contents": "Disease chapters: diagnosis, testing strategy, management, inheritance, counseling.",
         "manager": "University of Washington, on the NCBI Bookshelf.",
         "region": "International", "access": "Open", "url": "https://www.ncbi.nlm.nih.gov/books/NBK1116/"},

        # ---- Neurology-specific resources ----
        {"cat": "Neurology-specific resources", "name": "MDSGene", "acr": "MDSGene",
         "full": "MDSGene — movement-disorder genetics database",
         "desc": "Curated genotype–phenotype data for monogenic movement disorders (e.g., Parkinson's disease, dystonia, ataxia), systematically extracted from the published literature.",
         "contents": "Curated variants and clinical features for monogenic movement disorders.",
         "manager": "International Parkinson and Movement Disorder Society (MDS).",
         "region": "International", "access": "Open", "url": "https://www.mdsgene.org/"},
        {"cat": "Neurology-specific resources", "name": "MITOMAP", "acr": "MITOMAP",
         "full": "MITOMAP — human mitochondrial genome database",
         "desc": "Compendium of human mitochondrial DNA variation and its disease associations — relevant to mitochondrial neurological and metabolic disorders.",
         "contents": "mtDNA variants, disease associations, control-region polymorphisms.",
         "manager": "MITOMAP (Children's Hospital of Philadelphia).",
         "region": "International", "access": "Open", "url": "https://www.mitomap.org/"},

        # ---- Literature ----
        {"cat": "Literature & evidence", "name": "PubMed", "acr": "PubMed",
         "full": "PubMed — biomedical literature",
         "desc": "Primary index of biomedical literature; the underlying evidence source for variant and gene–disease curation in this and most clinical databases.",
         "contents": "Abstracts and citations across biomedical and life-sciences research.",
         "manager": "NCBI / National Library of Medicine (USA).",
         "region": "International", "access": "Open", "url": "https://pubmed.ncbi.nlm.nih.gov/"},
    ]

def _resource_card_html(r):
    access_color = {"Open": "#16a34a", "Free (academic)": "#0891b2", "Public + Licensed": "#b45309",
                    "Open (browser)": "#16a34a", "Licensed": "#dc2626", "Registered": "#7c3aed"}.get(r["access"], "#475569")
    pills = (f'<span style="display:inline-block;background:#eef6f9;border:1px solid #cfe6ee;color:#0b4f6c;'
             f'border-radius:999px;padding:0.15rem 0.55rem;font-size:0.74rem;font-weight:700;margin-left:0.3rem;">{r["region"]}</span>'
             f'<span style="display:inline-block;background:#ffffff;border:1px solid {access_color};color:{access_color};'
             f'border-radius:999px;padding:0.15rem 0.55rem;font-size:0.74rem;font-weight:700;margin-left:0.3rem;">{r["access"]}</span>')
    link = "" if r["url"] == "#" else f'<a class="hgdp-link-pill" href="{r["url"]}" target="_blank">Open resource ↗</a>'
    return (
        f'<div style="background:#ffffff;border:1px solid #d9e2ec;border-left:5px solid #007C92;'
        f'border-radius:14px;padding:1rem 1.1rem;margin-bottom:0.8rem;box-shadow:0 4px 14px rgba(15,23,42,0.05);">'
        f'<div style="display:flex;justify-content:space-between;align-items:baseline;gap:0.5rem;flex-wrap:wrap;">'
        f'<div style="font-size:1.08rem;font-weight:800;color:#102a43;">{r["name"]}'
        f'<span style="color:#007C92;font-size:0.9rem;font-weight:700;"> · {r["acr"]}</span></div>'
        f'<div>{pills}</div></div>'
        f'<div style="color:#486581;font-size:0.85rem;margin:0.1rem 0 0.55rem 0;font-style:italic;">{r["full"]}</div>'
        f'<div style="color:#1f2933;font-size:0.93rem;margin-bottom:0.55rem;line-height:1.45;">{r["desc"]}</div>'
        f'<div style="font-size:0.85rem;color:#334155;margin-bottom:0.2rem;"><b>Contents:</b> {r["contents"]}</div>'
        f'<div style="font-size:0.85rem;color:#334155;margin-bottom:0.6rem;"><b>Maintained by:</b> {r["manager"]}</div>'
        f'{link}</div>'
    )

def data_resources_page():
    st.subheader("📚 Data Resources — genetics & genomics database directory")
    st.markdown(
        "<div class='hgdp-subtitle' style='margin-bottom:0.6rem;'>An open directory of databases that support "
        "neurogenetic research in Pakistani and South Asian populations — from international clinical variant "
        "archives to regional reference genomes. Each entry notes what it holds, who maintains it, and links to the "
        "official resource.</div>", unsafe_allow_html=True)

    resources = get_data_resources()
    cats = ["Pakistan-focused resources", "Core clinical variant & disease databases",
            "Neurology-specific resources", "Literature & evidence"]

    c1, c2 = st.columns([2, 1])
    query = c1.text_input("Search resources (name, topic, organisation)", placeholder="e.g. HLA, mitochondrial, Pakistan, frequency")
    access_opts = sorted({r["access"] for r in resources})
    access_sel = c2.multiselect("Access type", access_opts, default=[])

    def matches(r):
        ok = True
        if query.strip():
            q = query.strip().lower()
            blob = " ".join([r["name"], r["acr"], r["full"], r["desc"], r["contents"], r["manager"], r["region"]]).lower()
            ok = ok and (q in blob)
        if access_sel:
            ok = ok and (r["access"] in access_sel)
        return ok

    shown = [r for r in resources if matches(r)]
    st.caption(f"Showing {len(shown)} of {len(resources)} resources.")
    for cat in cats:
        cat_items = [r for r in shown if r["cat"] == cat]
        if not cat_items:
            continue
        st.markdown(f"<div class='hgdp-section-label' style='margin-top:0.8rem;'>{cat} &nbsp;<span style='color:#94a3b8;font-weight:600;'>({len(cat_items)})</span></div>", unsafe_allow_html=True)
        for r in cat_items:
            st.markdown(_resource_card_html(r), unsafe_allow_html=True)

    st.markdown("---")
    st.caption("This directory is for orientation and links to third-party resources; please consult each database's own "
               "terms of use and the primary literature before relying on any record clinically.")

# ------------------- MAIN APP -------------------
def render_clingen_header():
    st.markdown(
        f"""
    <div class="hgdp-logo-header">
        <img class="hgdp-logo-img" src="{HEADER_LOGO_URI}"
             alt="Pakistan Neurogenomic Variant Repository (PNVR)"/>
    </div>
    """,
        unsafe_allow_html=True,
    )

def clingen_style_curation_cards(df):
    variant_classes = df["_display_variant_class"].replace("", np.nan).dropna().nunique()
    disease_count = df["_display_disease"].replace("", np.nan).dropna().nunique()
    ref_count = df["_display_reference"].replace("", np.nan).dropna().shape[0]
    ethnicity_count = df["_display_ethnicity"].replace("", np.nan).dropna().shape[0]
    html = f"""
    <div style="display: grid; grid-template-columns: repeat(4, minmax(180px, 1fr)); gap: 0.85rem; margin: 1rem 0;">
        <div style="background: #ffffff; border:1px solid #d9e2ec; border-top:4px solid #007C92; border-radius:14px; padding:1rem;"><div style="font-weight:800;">Gene-Disease Validity</div><div style="font-size:0.9rem;">Genes linked to neurological phenotypes</div><div style="font-size:1.35rem; font-weight:900; color:#007C92; margin-top:0.55rem;">{disease_count:,}</div></div>
        <div style="background: #ffffff; border:1px solid #d9e2ec; border-top:4px solid #007C92; border-radius:14px; padding:1rem;"><div style="font-weight:800;">Variant Pathogenicity</div><div style="font-size:0.9rem;">Mutation class, HGVS, etc.</div><div style="font-size:1.35rem; font-weight:900; color:#007C92; margin-top:0.55rem;">{variant_classes:,}</div></div>
        <div style="background: #ffffff; border:1px solid #d9e2ec; border-top:4px solid #007C92; border-radius:14px; padding:1rem;"><div style="font-weight:800;">Pakistan Population Context</div><div style="font-size:0.9rem;">Ethnicity, consanguinity</div><div style="font-size:1.35rem; font-weight:900; color:#007C92; margin-top:0.55rem;">{ethnicity_count:,}</div></div>
        <div style="background: #ffffff; border:1px solid #d9e2ec; border-top:4px solid #007C92; border-radius:14px; padding:1rem;"><div style="font-weight:800;">Literature Evidence</div><div style="font-size:0.9rem;">PubMed, OMIM, ClinVar</div><div style="font-size:1.35rem; font-weight:900; color:#007C92; margin-top:0.55rem;">{ref_count:,}</div></div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

# Main execution
# ---- Sidebar Logo ----
st.sidebar.markdown(
    f'<div style="text-align:center; padding:0.25rem 0 0.6rem 0;">'
    f'<img src="{LOGO_IMG_URI}" alt="Pakistan Neurogenomic Variant Repository (PNVR)" '
    f'style="width:100%; max-width:270px; height:auto; display:inline-block;"/></div>',
    unsafe_allow_html=True
)
st.sidebar.markdown("---")

render_clingen_header()
st.write("")

# Load mutation database from the default data path (data/neuro_mutations.csv)
df, meta = load_csv()
if df is None:
    st.info("No mutation database found yet. Place your CSV at data/neuro_mutations.csv (or neuro_mutations.parquet) next to this app.")
    st.stop()
st.sidebar.success(f"Loaded {len(df)} mutation records")

# Global search box
st.markdown('<div class="hgdp-search-panel"><div class="hgdp-section-label">🔍 Search the PNVR Atlas</div>', unsafe_allow_html=True)
global_query = st.text_input("Search gene, disease, variant, PMID, ethnicity, or chromosome", placeholder="Example: GRM1, ataxia, epilepsy, Pashtun, chr1, PMID", label_visibility="collapsed")
st.markdown("</div>", unsafe_allow_html=True)

# Sidebar filters
systems = sorted(df["_systems"].dropna().unique())
selected_systems = st.sidebar.multiselect("Affected system", systems, default=[])
variant_classes = sorted(df["_display_variant_class"].replace("", np.nan).dropna().unique())
selected_classes = st.sidebar.multiselect("Variant class", variant_classes, default=[])
ethnicities = sorted(df["_display_ethnicity"].replace("", np.nan).dropna().unique())
selected_ethnicities = st.sidebar.multiselect("Ethnicity / population", ethnicities, default=[])
diseases = sorted(df["_display_disease"].dropna().unique())
selected_diseases = st.sidebar.multiselect("Neurological disease", diseases, default=[])

# Apply global search and sidebar filters
filtered = df.copy()
if global_query.strip():
    q = global_query.strip().lower()
    q_alt = q.replace("chr", "") if q.startswith("chr") else q
    search_cols = ["_gene", "_chr", "_hgvs", "_display_variant_class", "_display_disease", "_inheritance", "_display_ethnicity", "_display_pmid", "_display_reference", "_display_paper_title"]
    mask = pd.Series(False, index=filtered.index)
    for col in search_cols:
        if col in filtered.columns:
            text = filtered[col].fillna("").astype(str).str.lower()
            mask = mask | text.str.contains(q, regex=False) | text.str.contains(q_alt, regex=False)
    filtered = filtered[mask]
if selected_systems:
    filtered = filtered[filtered["_systems"].isin(selected_systems)]
if selected_classes:
    filtered = filtered[filtered["_display_variant_class"].isin(selected_classes)]
if selected_ethnicities:
    filtered = filtered[filtered["_display_ethnicity"].isin(selected_ethnicities)]
if selected_diseases:
    filtered = filtered[filtered["_display_disease"].isin(selected_diseases)]

# Main navigation
section = st.sidebar.radio(
    "Dashboard section",
    ["Overview (Home)", "Chromosome tracks", "Genome-wide scatter", "Gene card", "Genome Browser", "Phenotype explorer", "Circos plot", "Data Resources", "Data quality"],
    key="dashboard_section"
)

if section == "Overview (Home)":
    # Alert banner
    st.markdown("""
    <div class="alert-banner">
        🔔 <strong>PNVR Update:</strong> 12 novel recessive ataxia genes identified in Punjabi families – March 2025
    </div>
    """, unsafe_allow_html=True)
    # KPI row
    pathogenic_mask = filtered["_display_variant_class"].str.contains("nonsense|frameshift|splice|stop", case=False, na=False)
    vus_mask = filtered["_display_variant_class"].str.contains("missense", case=False, na=False)
    consang_families = filtered[filtered["_consanguinity"].str.lower().isin(["yes", "first-cousin", "consanguineous"])].shape[0] if "_consanguinity" in filtered.columns else 0
    epilepsy_cases = filtered[filtered["_display_disease"].str.lower().str.contains("epilep", na=False)].shape[0]
    kpi_cols = st.columns(4)
    kpi_cols[0].markdown(f'<div class="kpi-card"><div class="kpi-value">{pathogenic_mask.sum()}</div><div class="kpi-label">🔬 Pathogenic</div></div>', unsafe_allow_html=True)
    kpi_cols[1].markdown(f'<div class="kpi-card"><div class="kpi-value">{vus_mask.sum()}</div><div class="kpi-label">⚠️ VUS</div></div>', unsafe_allow_html=True)
    kpi_cols[2].markdown(f'<div class="kpi-card"><div class="kpi-value">{consang_families}</div><div class="kpi-label">👨‍👩‍👧‍👦 Consanguineous Families</div></div>', unsafe_allow_html=True)
    kpi_cols[3].markdown(f'<div class="kpi-card"><div class="kpi-value">{epilepsy_cases}</div><div class="kpi-label">🧠 Epilepsy Cases</div></div>', unsafe_allow_html=True)

    # Left column (filters) and right column (table)
    left_col, right_col = st.columns([1.2, 2.8], gap="medium")
    with left_col:
        st.markdown("### 🧬 Filters & Context")
        selected_hpo = render_hpo_tree()
        if filtered["_onset"].notna().any():
            min_age = int(filtered["_onset"].min()) if not np.isnan(filtered["_onset"].min()) else 0
            max_age = int(filtered["_onset"].max()) if not np.isnan(filtered["_onset"].max()) else 80
            onset_range = st.slider("Age of onset (years)", min_age, max_age, (min_age, max_age))
        else:
            onset_range = (0, 100)
        render_region_map(filtered)
        inheritance_options = list(filtered["_inheritance"].dropna().unique()) if "_inheritance" in filtered.columns else []
        selected_inheritance = st.multiselect("Inheritance pattern", inheritance_options, default=[])
    # Apply additional filters
    filtered2 = filtered.copy()
    if selected_hpo:
        hpo_pattern = "|".join(selected_hpo)
        filtered2 = filtered2[filtered2["_display_disease"].str.lower().str.contains(hpo_pattern.lower(), na=False)]
    if filtered2["_onset"].notna().any():
        filtered2 = filtered2[(filtered2["_onset"] >= onset_range[0]) & (filtered2["_onset"] <= onset_range[1])]
    if selected_inheritance:
        filtered2 = filtered2[filtered2["_inheritance"].isin(selected_inheritance)]
    with right_col:
        st.markdown("### 📋 Variant Table (filtered)")
        table_df = filtered2[["_gene", "_hgvs", "_display_disease", "_onset", "_ethnicity", "_consanguinity", "_evidence_level"]].copy()
        table_df.columns = ["Gene", "Variant", "Disease", "Onset (yrs)", "Region", "Consanguinity", "Evidence Level"]
        table_df = table_df.fillna("—")
        st.dataframe(table_df, use_container_width=True, height=400)

    # Original overview content
    st.markdown("---")
    overview_panel(filtered)
    clingen_style_curation_cards(filtered)
    st.write("")
    st.subheader("Chromosome track preview (click a point for gene details)")
    selected_gene = chromosome_track_plot(filtered, CHR_ORDER, key="overview_track_select", label_mode="Auto", max_labels_per_chromosome=5)
    gene_from_state = selected_gene or st.session_state.get("selected_gene_from_track")
    if gene_from_state:
        st.write("")
        selected_gene_summary(filtered, gene_from_state)

    # Bottom section: publications, submit, export, API
    st.markdown("---")
    with st.expander("📚 Data Sources – Key Publications", expanded=True):
        papers = get_static_publications()
        for paper in papers:
            st.markdown(f"- **{paper['title']}**  \n  [{paper['date']}]({paper['link']})")
        st.caption("These publications have been used to source and curate the mutation data in this atlas.")
    st.markdown("**🔌 API Access**  \nREST API endpoints available on request.  \nContact: `curator@pnvr.pk`")

elif section == "Chromosome tracks":
    chromosome_tracks_page(filtered)
elif section == "Genome-wide scatter":
    genome_scatter(filtered)
elif section == "Gene card":
    gene_card(filtered)
elif section == "Genome Browser":
    genome_browser_page(filtered, genome_build="hg38")
elif section == "Phenotype explorer":
    phenotype_explorer(filtered)
elif section == "Circos plot":
    st.subheader("Multi‑ring Circos Plot by Variant Class")
    max_points = st.slider("Maximum points to show (performance)", 500, 5000, 2500, step=500)
    fig = circos_multiring_plot(filtered, max_points_total=max_points)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Could not generate multi‑ring Circos plot. Ensure your data contains at least one of the required variant classes (CNV, frameshift deletion/duplication, missense, nonsense, SNV, splicing).")
elif section == "Data Resources":
    data_resources_page()
elif section == "Data quality":
    quality_page(filtered, meta)