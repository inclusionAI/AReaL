%{
#include "ff.h"
#include "parse.h"
  
  /* default yywrap function - always treat EOF as an EOF  */
int fct_pddlwrap() { return 1; };

int gbracket_count = 0;

%}

a [Aa]
b [Bb]
c [Cc]
d [Dd]
e [Ee]
f [Ff]
g [Gg]
h [Hh]
i [Ii]
j [Jj]
k [Kk]
l [Ll]
m [Mm]
n [Nn]
o [Oo]
p [Pp]
q [Qq]
r [Rr]
s [Ss]
t [Tt]
u [Uu]
v [Vv]
w [Ww]
x [Xx]
y [Yy]
z [Zz]

%x COMMENT OVERREAD

%%

"("  { return(OPEN_PAREN); }

")"  {  return(CLOSE_PAREN); }

\([ \t]*{i}{n}"-"{p}{a}{c}{k}{a}{g}{e}  {  gbracket_count = 1;
 BEGIN OVERREAD; }

\([ \t]*":"{l}{e}{n}{g}{t}{h}  {  gbracket_count = 1;
 BEGIN OVERREAD; }

\([ \t]*":"{r}{e}{q}{u}{i}{r}{e}{m}{e}{n}{t}{s}  {  gbracket_count = 1;
 BEGIN OVERREAD; }

{d}{e}{f}{i}{n}{e}  {  return(DEFINE_TOK); }

{p}{r}{o}{b}{l}{e}{m}  {  return(PROBLEM_TOK); }

{s}{i}{t}{u}{a}{t}{i}{o}{n}  {  return(SITUATION_TOK); }

":"{s}{i}{t}{u}{a}{t}{i}{o}{n}  {  return(BSITUATION_TOK); }

":"{o}{b}{j}{e}{c}{t}{s}  {  return(OBJECTS_TOK); }

":"{g}{o}{a}{l}  {  return(GOAL_TOK); }

":"{m}{e}{t}{r}{i}{c}  {  return(METRIC_TOK); }

":"{i}{n}{i}{t}  {  return(INIT_TOK); }

":"{d}{o}{m}{a}{i}{n}  {  return(BDOMAIN_TOK); }

\([ \t]*":"{e}{x}{t}{e}{n}{d}{s}  {  gbracket_count = 1;
 BEGIN OVERREAD; }

{a}{n}{d}  {  return(AND_TOK); }

{i}{m}{p}{l}{y} {  return(IMPLY_TOK); }

{o}{r} {  return(OR_TOK); }

{f}{o}{r}{a}{l}{l} {  return(FORALL_TOK); }

{e}{x}{i}{s}{t}{s} {  return(EXISTS_TOK); }

{n}{o}{t}  {  return(NOT_TOK); }

"<"  {  return(LE_TOK); }

"<="  {  return(LEQ_TOK); }

"="  {  return(EQ_TOK); }

">="  {  return(GEQ_TOK); }

">"  {  return(GE_TOK); }

"-"  {  return(MINUS_TOK); }

"+"  {  return(AD_TOK); }

"*"  {  return(MU_TOK); }

"/"  {  return(DI_TOK); }

:?[a-zA-Z][a-zA-Z0-9\-_]* { strupcase( yytext );  
  strcpy(yylval.string, yytext ); return(NAME); }

\?[a-zA-Z][a-zA-Z0-9\-_\[\]]* {strupcase( yytext );
 strcpy(yylval.string, yytext); return(VARIABLE); }

"-"?[0-9]*[.]?[0-9]* { strcpy(yylval.string, yytext); return(NUM);}

"-"[ \t]*"("[ \t]*{e}{i}{t}{h}{e}{r} { return(EITHER_TOK); }

\;(.)*\n  {  lineno++; } 
\;(.)*  {  /* this will hold only in files that end with
		   a comment but no linefeed */ } 

<COMMENT>(.^\")*\n    {  lineno++; }  ;

<INITIAL>\" { BEGIN COMMENT;}

<COMMENT>\" { BEGIN INITIAL;}

\n    {  lineno++; } 

<OVERREAD>(.^\(\))*\n  {  lineno++; }

<OVERREAD>[^\(\)]  {  }

<OVERREAD>\(  {  gbracket_count++; }

<OVERREAD>\)  {  gbracket_count--; 
  if (!gbracket_count) BEGIN INITIAL; }

. {}
%%
