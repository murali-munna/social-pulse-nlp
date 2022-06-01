import pandas as pd
import streamlit as st
import re
import pickle
import os
import html
from htbuilder import H, HtmlElement, styles
from htbuilder.units import unit

########################################################################################################################

filename = '273_sentimentAppend.csv'
quantity = 10
keywordMarker = 'KW'
posColor = '#0068c9'
negColor = '#0068c9'
negProbThresh = .95

# path = './data/'
# df = pd.read_pickle(os.path.join(path,'df.pkl'))
# df['votes'] = df['ups'].fillna(df['favorite_count'])
# df = df[['text','sentiment_prediction','sentiment_probability','brand','stream','votes']]
# kwDict = pickle.load(open(os.path.join(path,'kw_yake.pkl'), 'rb'))


########################################################################################################################

def annotated_text(args, color=None):
    """Writes text with annotations into your Streamlit app.

    Parameters
    ----------
    *args : str, tuple or htbuilder.HtmlElement
        Arguments can be:
        - strings, to draw the string as-is on the screen.
        - tuples of the form (main_text, annotation_text, background, color) where
          background and foreground colors are optional and should be an CSS-valid string such as
          "#aabbcc" or "rgb(10, 20, 30)"
        - HtmlElement objects in case you want to customize the annotations further. In particular,
          you can import the `annotation()` function from this module to easily produce annotations
          whose CSS you can customize via keyword arguments.

    Examples
    --------

    >>> annotated_text(
    ...     "This ",
    ...     ("is", "verb", "#8ef"),
    ...     " some ",
    ...     ("annotated", "adj", "#faa"),
    ...     ("text", "noun", "#afa"),
    ...     " for those of ",
    ...     ("you", "pronoun", "#fea"),
    ...     " who ",
    ...     ("like", "verb", "#8ef"),
    ...     " this sort of ",
    ...     ("thing", "noun", "#afa"),
    ... )

    >>> annotated_text(
    ...     "Hello ",
    ...     annotation("world!", "noun", color="#8ef", border="1px dashed red"),
    ... )

    """
    st.markdown(
        get_annotated_html(args, color),
        unsafe_allow_html=True,
    )

########################################################################################################################


# Only works in 3.7+: from htbuilder import div, span
div = H.div
span = H.span

# Only works in 3.7+: from htbuilder.units import px, rem, em
px = unit.px
rem = unit.rem
em = unit.em

# Colors from the Streamlit palette.
# These are red-70, orange-70, ..., violet-70, gray-70.
PALETTE = [
    "#ff4b4b",
    "#ffa421",
    "#ffe312",
    "#21c354",
    "#00d4b1",
    "#00c0f2",
    "#1c83e1",
    "#803df5",
    "#808495",
]

OPACITIES = [
    "33", "66",
]

def annotation(body, label="", background=None, color=None, **style):
    """Build an HtmlElement span object with the given body and annotation label.

    The end result will look something like this:

        [body | label]

    Parameters
    ----------
    body : string
        The string to put in the "body" part of the annotation.
    label : string
        The string to put in the "label" part of the annotation.
    background : string or None
        The color to use for the background "chip" containing this annotation.
        If None, will use a random color based on the label.
    color : string or None
        The color to use for the body and label text.
        If None, will use the document's default text color.
    style : dict
        Any CSS you want to apply to the containing "chip". This is useful for things like


    Examples
    --------

    Produce a simple annotation with default colors:

    >>> annotation("apple", "fruit")

    Produce an annotation with custom colors:

    >>> annotation("apple", "fruit", background="#FF0", color="black")

    Produce an annotation with crazy CSS:

    >>> annotation("apple", "fruit", background="#FF0", border="1px dashed red")

    """

    color_style = {}

    if color:
        color_style['color'] = color

    if not background:
        label_sum = sum(ord(c) for c in label)
        background_color = PALETTE[label_sum % len(PALETTE)]
        background_opacity = OPACITIES[label_sum % len(OPACITIES)]
        background = background_color + background_opacity

    return (
        span(
            style=styles(
                background=background,
                border_radius=rem(0.33),
                padding=(rem(0.125), rem(0.5)),
                overflow="hidden",
                **color_style,
                **style,
            ))(

            html.escape(body),

            span(
                style=styles(
                    padding_left=rem(0.5),
                    text_transform="uppercase",
                ))(
                span(
                    style=styles(
                        font_size=em(0.67),
                        opacity=0.5,
                    ))(
                    html.escape(label),
                ),
            ),
        )
    )


def get_annotated_html(args, color):
    """Writes text with annotations into an HTML string.

    Parameters
    ----------
    *args : see annotated_text()

    Returns
    -------
    str
        An HTML string.
    """

    out = div()

    for arg in args:
        if isinstance(arg, str):
            out(html.escape(arg))

        elif isinstance(arg, HtmlElement):
            out(arg)

        elif isinstance(arg, tuple):
            out(annotation(*arg, color))

        else:
            raise Exception("Oh noes!")

    return str(out)


########################################################################################################################



def postExtremeComments(data, keywords, color, extreme):

    with st.expander(f"Most {extreme} Comments", expanded=True):
        AAList = []
        for _, row in data.iterrows():
            text = row["text"]
            annotations = []
            for keyword in keywords:
                split = re.split(keyword, text, flags=re.IGNORECASE)
                if len(split) >= 2:
                    for i in range(len(split) - 1):
                        splitLenSum = sum([len(split[j]) for j in range(i + 1)])
                        annotations.append((splitLenSum + len(keyword) * i, keyword))

            annotations.sort(key=lambda y: y[0])
            annotateArgs = ['📝']
            strPointer = 0

            if annotations == []:
                AAList.append([text])
                continue

            if annotations[0][0] != 0:
                annotateArgs.append(text[0:annotations[0][0] - 1] + ' ')
                strPointer = len(annotateArgs[-1])
            for idx, obj in enumerate(annotations):
                if strPointer != obj[0]:
                    annotateArgs.append(text[strPointer:obj[0]])
                    strPointer += len(annotateArgs[-1])
                annotateArgs.append((obj[1], keywordMarker))
                strPointer += len(annotations[idx][1])

            if strPointer != len(text):
                annotateArgs.append(text[strPointer:])

            annotateArgs.append(' (votes: {})'.format(int(row['votes'])))
            # annotateArgs.append('\n')
            AAList.append(annotateArgs)

        for i, annotatedArgs in enumerate(AAList):
            annotated_text(annotatedArgs, color=color)
            if i!=len(AAList)-1:
                st.markdown("""---""")



# show_posts('Ferrari','Twitter',df,kwDict)