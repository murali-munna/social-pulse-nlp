import streamlit as st

BACKGROUND_COLOR = 'white'
COLOR = 'black'

def set_page_container_style(
        max_width: int = 1100, max_width_100_percent: bool = False,
        padding_top: int = 1, padding_right: int = 10, padding_left: int = 1, padding_bottom: int = 10,
        color: str = COLOR, background_color: str = BACKGROUND_COLOR,
    ):
        """
        It sets the style of the page container

        Args:
          max_width (int): The maximum width of the page. Defaults to 1100
          max_width_100_percent (bool): If True, the max-width of the page will be 100%. If False, the max-width will be 1100px.
        Defaults to False
          padding_top (int): int = 1, padding_right: int = 10, padding_left: int = 1, padding_bottom: int = 10,. Defaults to 1
          padding_right (int): int = 10, padding_left: int = 1, padding_bottom: int = 10,. Defaults to 10
          padding_left (int): int = 1, padding_right: int = 10, padding_top: int = 1, padding_bottom: int = 10,. Defaults to 1
          padding_bottom (int): int = 10,. Defaults to 10
          color (str): The color of the text.
          background_color (str): The background color of the page.
        """
        if max_width_100_percent:
            max_width_str = f'max-width: 100%;'
        else:
            max_width_str = f'max-width: {max_width}px;'
        st.markdown(
            f'''
            <style>
                .reportview-container .css-1lcbmhc .css-1outpf7 {{
                    padding-top: 35px;
                }}
                .reportview-container .main .block-container {{
                    {max_width_str}
                    padding-top: {padding_top}rem;
                    padding-right: {padding_right}rem;
                    padding-left: {padding_left}rem;
                    padding-bottom: {padding_bottom}rem;
                }}
                .reportview-container .main {{
                    color: {color};
                    background-color: {background_color};
                }}
            </style>
            ''',
            unsafe_allow_html=True,
        )