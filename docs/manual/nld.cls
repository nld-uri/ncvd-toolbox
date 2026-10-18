%% nld.cls -- NLD documents: reports, proposals, letters
%% Nonlinear Dynamics Laboratory, University of Rhode Island
%%
%% \documentclass[report]{nld}    cover page + numbered sections (default)
%% \documentclass[proposal]{nld}  no cover, running header, page prefix
%% \documentclass[letter]{nld}    letterhead on page 1
%% Compile with XeLaTeX or LuaLaTeX (recommended) or pdfLaTeX.
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{nld}[2026/10/10 v0.1 NLD documents]

\newif\ifnld@report \newif\ifnld@proposal \newif\ifnld@letter
\nld@reporttrue
\DeclareOption{report}{\nld@reporttrue\nld@proposalfalse\nld@letterfalse}
\DeclareOption{proposal}{\nld@reportfalse\nld@proposaltrue\nld@letterfalse}
\DeclareOption{letter}{\nld@reportfalse\nld@proposalfalse\nld@lettertrue}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessOptions\relax
\LoadClass[11pt,letterpaper]{article}

\RequirePackage{nld-style}
\RequirePackage{graphicx}
\RequirePackage[letterpaper,left=1in,right=1in,top=1in,bottom=1in,headsep=0.3in,footskip=0.45in]{geometry}
\RequirePackage{fancyhdr}
\RequirePackage{titlesec}
\RequirePackage{enumitem}
\RequirePackage{booktabs}
\RequirePackage[font={sf,small},labelfont={color=nldink,bf},textfont={color=nldmuted},labelsep=quad]{caption}
\RequirePackage{tikz}
\RequirePackage[hidelinks]{hyperref}
\hypersetup{colorlinks=true,linkcolor=nldcopperdeep,urlcolor=nldcopperdeep,citecolor=nldcopperdeep}

% Logo directory: override with \renewcommand{\nldlogodir}{path/} if needed
\newcommand{\nldlogodir}{logos/}

\color{nldink}
\linespread{1.08}
\setlength{\parindent}{0pt}
\setlength{\parskip}{0.6\baselineskip}
\setlist{itemsep=0.2em,topsep=0.3em,leftmargin=1.4em}
\setlist[itemize]{label={\color{nldcopper}\rule[0.25ex]{0.45em}{0.45em}}}

% ---------------------------------------------------------------- headings
\titleformat{\section}{\sffamily\bfseries\Large\color{nldink}}{\textcolor{nldcopper}{\thesection}}{0.9em}{}
\titleformat{\subsection}{\sffamily\bfseries\large\color{nldink}}{\textcolor{nldcopper}{\thesubsection}}{0.9em}{}
\titleformat{\subsubsection}{\sffamily\bfseries\normalsize\color{nldink}}{\textcolor{nldcopper}{\thesubsubsection}}{0.9em}{}
\titlespacing*{\section}{0pt}{1.6em}{0.5em}
\titlespacing*{\subsection}{0pt}{1.2em}{0.3em}
\titlespacing*{\subsubsection}{0pt}{1.0em}{0.2em}

% ---------------------------------------------------------------- metadata
\newcommand{\nld@reportnumber}{}\newcommand{\reportnumber}[1]{\renewcommand{\nld@reportnumber}{#1}}
\newcommand{\nld@subtitle}{}\newcommand{\subtitle}[1]{\renewcommand{\nld@subtitle}{#1}}
\newcommand{\nld@runninghead}{}\newcommand{\runninghead}[1]{\renewcommand{\nld@runninghead}{#1}}
\newcommand{\nld@pageprefix}{}\newcommand{\pageprefix}[1]{\renewcommand{\nld@pageprefix}{#1}}
\newcommand{\nld@affiliation}{Nonlinear Dynamics Laboratory \textperiodcentered{} University of Rhode Island \textperiodcentered{} Kingston, RI}
\newcommand{\affiliationline}[1]{\renewcommand{\nld@affiliation}{#1}}

% Lab tagline on the report cover; \tagline{}{} to change, \tagline{}{} with empty first argument to hide
\newcommand{\nld@tagline}{When amplitude changes the story}
\newcommand{\nld@taglinesub}{In nonlinear structures, amplitude changes frequency, mode shape and stability.}
\newcommand{\tagline}[2]{\renewcommand{\nld@tagline}{#1}\renewcommand{\nld@taglinesub}{#2}}

% ---------------------------------------------------------------- page styles
\renewcommand{\headrulewidth}{0pt}
\fancypagestyle{nldplain}{%
  \fancyhf{}%
  \fancyhead[L]{\sffamily\scriptsize\color{nldmuted}\nld@runninghead}%
  \fancyhead[R]{\sffamily\scriptsize\color{nldmuted}NLD \textperiodcentered{} URI}%
  \fancyfoot[C]{\sffamily\scriptsize\color{nldmuted}\nld@pageprefix\thepage}%
  \renewcommand{\headrule}{\color{nldhairline}\hrule height 0.5pt}%
  \renewcommand{\headrulewidth}{0.5pt}}
\pagestyle{nldplain}

% ---------------------------------------------------------------- report cover
\renewcommand{\maketitle}{%
  \thispagestyle{empty}%
  \begin{tikzpicture}[remember picture,overlay]
    \fill[nldnavy] (current page.north west) rectangle ([yshift=-2.35in]current page.north east);
    \begin{scope}
      \clip (current page.north west) rectangle ([yshift=-2.35in]current page.north east);
      \node[anchor=center,inner sep=0] at ([xshift=-1.25in,yshift=-0.95in]current page.north east)
        {\includegraphics[width=3.1in]{\nldlogodir nld-mark-on-navy.pdf}};
    \end{scope}
    \node[anchor=north west,inner sep=0] at ([xshift=1in,yshift=-0.85in]current page.north west)
      {\includegraphics[height=0.55in]{\nldlogodir nld-lockup-horizontal-on-navy.pdf}};
    \ifx\nld@tagline\empty\else
    \node[anchor=north west,inner sep=0,text width=4.3in,align=left] at ([xshift=1in,yshift=-1.62in]current page.north west)
      {{\sffamily\bfseries\normalsize\color{nldpaper}\nld@tagline\par}\vspace{0.25em}%
       {\sffamily\footnotesize\color{nldmutednavy}\hyphenpenalty=10000\exhyphenpenalty=10000 \nld@taglinesub\par}};
    \fi
    \draw[nldhairline,line width=0.5pt] ([xshift=1in,yshift=0.75in]current page.south west) -- ([xshift=-1in,yshift=0.75in]current page.south east);
    \node[anchor=north west,inner sep=0,text=nldmuted,font=\sffamily\scriptsize] at ([xshift=1in,yshift=0.62in]current page.south west) {\nld@affiliation};
    \node[anchor=north west,inner sep=0,text width=6.5in,align=left] at ([xshift=1in,yshift=-3.3in]current page.north west) {%
      \ifx\nld@reportnumber\empty\else\nldeyebrow{\nld@reportnumber}\par\vspace{0.8em}\fi
      {\sffamily\bfseries\fontsize{30}{35}\selectfont\color{nldink}\@title\par}%
      \ifx\nld@subtitle\empty\else\vspace{0.6em}{\sffamily\Large\color{nldmuted}\nld@subtitle\par}\fi
      \vspace{2.2em}%
      {\sffamily\normalsize\color{nldink}\@author\par\vspace{0.2em}\color{nldmuted}\@date\par}};
  \end{tikzpicture}%
  \null
  \newpage}

% ---------------------------------------------------------------- letter
\newcommand{\nld@sender}{David Chelidze}
\newcommand{\nld@sendertitle}{Professor and Director}
\newcommand{\nld@address}{Department of Mechanical, Industrial\\and Systems Engineering\\University of Rhode Island\\Kingston, RI 02881}
\newcommand{\nld@contact}{chelidze@uri.edu \textperiodcentered{} (401) 874-2356}
\newcommand{\nld@web}{web.uri.edu/nld}
\newcommand{\sender}[2]{\renewcommand{\nld@sender}{#1}\renewcommand{\nld@sendertitle}{#2}}
\newcommand{\senderaddress}[1]{\renewcommand{\nld@address}{#1}}
\newcommand{\sendercontact}[1]{\renewcommand{\nld@contact}{#1}}
\fancypagestyle{nldletterhead}{%
  \fancyhf{}\renewcommand{\headrulewidth}{0pt}\renewcommand{\headrule}{}%
  \fancyfoot[L]{\sffamily\scriptsize\color{nldmuted}\nld@web}%
  \fancyfoot[R]{\sffamily\scriptsize\color{nldmuted}\nld@contact}%
  \renewcommand{\footrule}{\color{nldhairline}\hrule height 0.5pt\vspace{4pt}}%
  \renewcommand{\footrulewidth}{0.5pt}}
\fancypagestyle{nldletterrest}{%
  \fancyhf{}\renewcommand{\headrulewidth}{0pt}\renewcommand{\headrule}{}%
  \fancyfoot[R]{\sffamily\scriptsize\color{nldmuted}\thepage}}
\newenvironment{nldletter}[1]{%
  \thispagestyle{nldletterhead}\pagestyle{nldletterrest}%
  \vspace*{-0.45in}%
  \noindent\begin{minipage}[t]{0.5\textwidth}\vspace{0pt}%
    \includegraphics[height=0.58in]{\nldlogodir nld-lockup-horizontal.pdf}\end{minipage}%
  \hfill\begin{minipage}[t]{0.45\textwidth}\vspace{2pt}\raggedleft
    {\sffamily\scriptsize\color{nldmuted}\nld@address\par}\end{minipage}\par
  \vspace{0.9em}{\color{nldcopper}\rule{0.67in}{3pt}}\par\vspace{1.4em}%
  \raggedright
  \@date\par\vspace{0.6em}%
  #1\par\vspace{0.6em}}{%
  \par\vspace{1.2em}Sincerely,\par\vspace{3.2em}%
  \nld@sender\par\vspace{-0.6em}{\sffamily\small\color{nldmuted}\nld@sendertitle}\par}

\ifnld@proposal
  \pageprefix{}\runninghead{Project Description}
\fi
\ifnld@letter
  \pagestyle{nldletterrest}
\fi

\endinput
