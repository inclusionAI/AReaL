/*********************************************************************
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
 * 
 *********************************************************************/


/*********************************************************************
 * File: output.c
 * Description: printing info out
 *
 * Author: Joerg Hoffmann
 *
 *********************************************************************/ 





#include "ff.h"

#include "output.h"







/* parsing
 */







void print_FactList( FactList *list, char *sepf, char *sept )

{

  FactList *i_list;
  TokenList *i_tl;
    
  if ( list ) {
    i_tl = list->item;
    if (NULL == i_tl || NULL == i_tl->item) {
      printf("empty");
    } else {
      printf("%s", i_tl->item);
      i_tl = i_tl->next;
    }
    
    while (NULL != i_tl) {
      if (NULL != i_tl->item) {
	printf("%s%s", sept, i_tl->item);
      }
      i_tl = i_tl->next;
    }
    
    for ( i_list = list->next; i_list; i_list = i_list->next ) {
      printf("%s", sepf);
      i_tl = i_list->item;
      if (NULL == i_tl || NULL == i_tl->item) {
	printf("empty");
      } else {
	printf("%s", i_tl->item);
	i_tl = i_tl->next;
      }
      
      while (NULL != i_tl) {
	if (NULL != i_tl->item) {
	  printf("%s%s", sept, i_tl->item);
	}
	i_tl = i_tl->next;
      }
    }
  }

}



void print_hidden_TokenList( TokenList *list, char *sep )

{

  TokenList *i_tl;

  i_tl = list;
  if (NULL!=i_tl) {
    printf("%s", i_tl->item);
    i_tl = i_tl->next;
  } else {
    printf("empty");
  }
  
  while (NULL != i_tl) {
    printf("%s%s", sep, i_tl->item);
    i_tl = i_tl->next;
  }
  
}



void print_indent( int indent )

{

  int i;
  for (i=0;i<indent;i++) {
    printf(" ");
  }

}



void print_ParseExpNode( ParseExpNode *n )

{

  if ( !n ) return;

  switch ( n->connective) {
  case AD:
    printf("(+ ");
    print_ParseExpNode( n->leftson );
    print_ParseExpNode( n->rightson );
    printf(")");
    break;
  case SU:
    printf("(- ");
    print_ParseExpNode( n->leftson );
    print_ParseExpNode( n->rightson );
    printf(")");
    break;
  case MU:
    printf("(* ");
    print_ParseExpNode( n->leftson );
    print_ParseExpNode( n->rightson );
    printf(")");
    break;
  case DI:
    printf("(/ ");
    print_ParseExpNode( n->leftson );
    print_ParseExpNode( n->rightson );
    printf(")");
    break;
  case MINUS:
    printf("(- ");
    print_ParseExpNode( n->leftson );
    printf(")");
    break;
  case NUMBER:
    printf("%s", n->atom->item);
    break;
  case FHEAD:
    printf("(");
    print_hidden_TokenList(n->atom, " ");
    printf(")");
    break;
  default:
    printf("\n\nprint Parseexpnode: wrong specifier %d",
	   n->connective);
  }

}



void print_PlNode( PlNode *plnode, int indent )

{

  PlNode *i_son;

  if ( !plnode ) {
    printf("none\n");
    return;
  }
  
  switch (plnode->connective) {
  case ALL: 
    printf("ALL %s : %s\n", plnode->atom->item,
	    plnode->atom->next->item);
    print_indent(indent);
    printf("(   ");
    print_PlNode(plnode->sons,indent+4);
    print_indent(indent);
    printf(")\n");
    break;
  case EX:
    printf("EX  %s : %s\n", plnode->atom->item,
	    plnode->atom->next->item);
    print_indent(indent);
    printf("(   ");
    print_PlNode(plnode->sons,indent+4);
    print_indent(indent);
    printf(")\n");
    break;
  case AND: 
    printf("A(  ");
    print_PlNode(plnode->sons, indent+4);
    if ( plnode->sons ) {
      for ( i_son = plnode->sons->next; i_son!=NULL; i_son = i_son->next ) {
	print_indent(indent);
	printf("AND ");
	print_PlNode(i_son,indent+4);
      }
    }
    print_indent(indent);      
    printf(")\n");
    break;
  case OR:  
    printf("O(  ");
    print_PlNode(plnode->sons, indent+4);
    for ( i_son = plnode->sons->next; i_son!=NULL; i_son = i_son->next ) {
      print_indent(indent);
      printf("OR ");
      print_PlNode(i_son,indent+4);
    }
    print_indent(indent);      
    printf(")\n");
    break;
  case WHEN:
    printf("IF   ");
    print_PlNode(plnode->sons,indent+5);
    print_indent(indent);
    printf("THEN ");
    print_PlNode(plnode->sons->next,indent+5);
    print_indent(indent);
    printf("ENDIF\n");
    break;
  case NOT:
    if (ATOM==plnode->sons->connective) {
      printf("NOT ");
      print_PlNode(plnode->sons,indent+4);
    } else {
      printf("NOT(");
      print_PlNode(plnode->sons,indent+4);
      print_indent(indent+3);
      printf(")\n");
    }
    break;
  case ATOM:
    printf("(");
    print_hidden_TokenList(plnode->atom, " ");
    printf(")\n");
    break;
  case TRU:
     printf("(TRUE)\n");
     break;
  case FAL:
     printf("(FALSE)\n");
     break;   
  case COMP:
    switch (plnode->comp) {
    case LE:
      printf("(< ");
      break;
    case LEQ:
      printf("(<= ");
      break;
    case EQ:
      printf("(= ");
      break;
    case GEQ:
      printf("(>= ");
      break;
    case GE:
      printf("(> ");
      break;
    default:
      printf("\n\nillegal comp in parse tree!\n\n");
      exit( 1 );
    }
    print_ParseExpNode( plnode->lh );
    print_ParseExpNode( plnode->rh );
    printf(")\n");
    break;
  case NEF:
    switch (plnode->neft) {
    case ASSIGN:
      printf("(assign ");
      break;
    case SCALE_UP:
      printf("(scale-up ");
      break;
    case SCALE_DOWN:
      printf("(scale-down ");
      break;
    case INCREASE:
      printf("(increase ");
      break;
    case DECREASE:
      printf("(decrease ");
      break;
    }
    print_ParseExpNode( plnode->lh );
    print_ParseExpNode( plnode->rh );
    printf(")\n");
    break;
  default:
    printf("\n***** ERROR ****");
    printf("\nprint_plnode: %d > Wrong Node specifier\n", plnode->connective);
    exit(1);
  }     

} 



void print_plops( PlOperator *plop )

{

  PlOperator *i_plop;
  int count = 0;

  if ( !plop ) {
    printf("none\n");
  }

  for ( i_plop = plop; i_plop!=NULL; i_plop = i_plop->next ) {
    printf("\n");
    if ( i_plop->axiom ) printf("AXIOM-");
    printf("OPERATOR ");
    printf("%s", i_plop->name);
    printf("\nparameters: (%d real)\n", i_plop->number_of_real_params);
    print_FactList ( i_plop->params, "\n", " : ");
    printf("\n\npreconditions:\n");
    print_PlNode(i_plop->preconds, 0);
    printf("effects:\n");
    print_PlNode(i_plop->effects, 0);
    printf("\n-----\n");
    count++;
  }
  printf("\nAnzahl der Operatoren: %d\n", count);

}



void print_ExpNode( ExpNode *n )

{

  if ( !n ) return;

  switch ( n->connective) {
  case AD:
    printf("(+ ");
    print_ExpNode( n->leftson );
    print_ExpNode( n->rightson );
    printf(")");
    break;
  case SU:
    printf("(- ");
    print_ExpNode( n->leftson );
    print_ExpNode( n->rightson );
    printf(")");
    break;
  case MU:
    printf("(* ");
    print_ExpNode( n->leftson );
    print_ExpNode( n->rightson );
    printf(")");
    break;
  case DI:
    printf("(/ ");
    print_ExpNode( n->leftson );
    print_ExpNode( n->rightson );
    printf(")");
    break;
  case MINUS:
    printf("(- ");
    print_ExpNode( n->son );
    printf(")");
    break;
  case NUMBER:
    printf("%.2f", n->value);
    break;
  case FHEAD:
    if ( n->fluent ) {
      print_Fluent( n->fluent );
    } else {
      if ( n->fl >= 0 ) {
	printf(" %.2f*", n->c);
	print_fl_name( n->fl );
      } else {
	printf("[UNDEF]");
      }
    }
    break;
  default:
    printf("\n\nprint Expnode: wrong specifier %d",
	   n->connective);
  }

}



void print_Wff( WffNode *n, int indent )

{

  WffNode *i;

  if ( !n ) {
    printf("none\n");
    return;
  }
  
  switch (n->connective) {
  case ALL: 
    printf("ALL x%d (%s): %s\n", n->var, n->var_name,
	    gtype_names[n->var_type]);
    print_indent(indent);
    printf("(   ");
    print_Wff(n->son,indent+4);
    print_indent(indent);
    printf(")\n");
    break;
  case EX:
    printf("EX  x%d (%s) : %s\n",  n->var, n->var_name,
	    gtype_names[n->var_type]);
    print_indent(indent);
    printf("(   ");
    print_Wff(n->son,indent+4);
    print_indent(indent);
    printf(")\n");
    break;
  case AND: 
    printf("A(  ");
    print_Wff(n->sons, indent+4);
    if ( n->sons ) {
      for ( i = n->sons->next; i!=NULL; i = i->next ) {
	if ( !i->prev ) {
	  printf("\nprev in AND not correctly set!\n\n");
	  exit( 1 );
	}
	print_indent(indent);
	printf("AND ");
	print_Wff(i,indent+4);
      }
    }
    print_indent(indent);      
    printf(")\n");
    break;
  case OR:  
    printf("O(  ");
    print_Wff(n->sons, indent+4);
    for ( i = n->sons->next; i!=NULL; i = i->next ) {
      print_indent(indent);
      printf("OR ");
      print_Wff(i,indent+4);
    }
    print_indent(indent);      
    printf(")\n");
    break;
  case NOT:
    if (ATOM==n->son->connective) {
      printf("NOT ");
      print_Wff(n->son,indent+4);
    } else {
      printf("NOT(");
      print_Wff(n->son,indent+4);
      print_indent(indent+3);
      printf(")\n");
    }
    break;
  case ATOM:
    print_Fact(n->fact);
    if ( n->NOT_p != -1 ) printf(" - translation NOT");
    printf("\n");
    break;
  case TRU:
     printf("(TRUE)\n");
     break;
  case FAL:
     printf("(FALSE)\n");
     break;   
  case COMP:
    switch (n->comp) {
    case LE:
      printf("(< ");
      break;
    case LEQ:
      printf("(<= ");
      break;
    case EQ:
      printf("(= ");
      break;
    case GEQ:
      printf("(>= ");
      break;
    case GE:
      printf("(> ");
      break;
    default:
      printf("\nwrong comparator of Expnodes in WFF %d\n\n", n->comp);
      exit( 1 );
    }
    print_ExpNode( n->lh );
    print_ExpNode( n->rh );
    printf(")\n");
    break;
  default:
    printf("\n***** ERROR ****");
    printf("\nprint_Wff: %d > Wrong Node specifier\n", n->connective);
    exit(1);
  }     

} 



void print_Operator( Operator *o )

{

  Effect *e;
  Literal *l;
  NumericEffect *ne;
  int i, m = 0;

  printf("\n\n----------------Operator %s, axiom %d, translated form, step 1--------------\n", 
	 o->name, o->axiom);

  for ( i = 0; i < o->num_vars; i++ ) {
    printf("\nx%d (%s) of type %s, removed ? %s",
	   i, o->var_names[i], gtype_names[o->var_types[i]],
	   o->removed[i] ? "YES" : "NO");
  }
  printf("\ntotal params %d, real params %d\n", 
	 o->num_vars, o->number_of_real_params);

  printf("\nPreconds:\n");
  print_Wff( o->preconds, 0 );

  printf("\n\nEffects:");
  for ( e = o->effects; e; e = e->next ) {
    printf("\n\neffect %d, parameters %d", m++, e->num_vars);

    for ( i = 0; i < e->num_vars; i++ ) {
      printf("\nx%d (%s) of type %s",
	     o->num_vars + i, e->var_names[i], gtype_names[e->var_types[i]]);
    }
    printf("\nConditions\n");
    print_Wff( e->conditions, 0 );
    printf("\nEffect Literals");
    for ( l = e->effects; l; l = l->next ) {
      if ( l->negated ) {
	printf("\nNOT ");
      } else {
	printf("\n");
      }
      print_Fact( &(l->fact) );
    }
    printf("\nNumeric Effects");
    for ( ne = e->numeric_effects; ne; ne = ne->next ) {
      switch ( ne->neft ) {
      case ASSIGN:
	printf("\nassign ");
	break;
      case SCALE_UP:
	printf("\nscale-up ");
	break;
      case SCALE_DOWN:
	printf("\nscale-down ");
	break;
      case INCREASE:
	printf("\nincrease ");
	break;
      case DECREASE:
	printf("\ndecrease ");
	break;
      default:
	printf("\n\nprint effect: illegal neft %d\n\n", ne->neft);
	exit( 1 );
      }
      print_Fluent( &(ne->fluent) );
      print_ExpNode( ne->rh );
    }
  }

}



void print_NormOperator( NormOperator *o )

{

  NormEffect *e;
  int i, m;

  printf("\n\n----------------Operator %s, normalized form--------------\n", 
	 o->operator->name);

  for ( i = 0; i < o->num_vars; i++ ) {
    printf("\nx%d of type ", i);
    print_type( o->var_types[i] );
  }
  printf("\n\n%d vars removed from original operator:",
	 o->num_removed_vars);
  for ( i = 0; i < o->num_removed_vars; i++ ) {
    m = o->removed_vars[i];
    printf("\nx%d (%s) of type %s, type constraint ", m, o->operator->var_names[m], 
	   gtype_names[o->operator->var_types[m]]);
    print_type( o->type_removed_vars[i] );
  }

  printf("\nPreconds:\n");
  for ( i = 0; i < o->num_preconds; i++ ) {
    print_Fact( &(o->preconds[i]) );
    printf("\n");
  }
  for ( i = 0; i < o->num_numeric_preconds; i++ ) {
    switch ( o->numeric_preconds_comp[i] ) {
    case LE:
      printf("(< ");
      break;
    case LEQ:
      printf("(<= ");
      break;
    case EQ:
      printf("(= ");
      break;
    case GEQ:
      printf("(>= ");
      break;
    case GE:
      printf("(> ");
      break;
    default:
      printf("\nwrong comparator of Expnodes in normpre %d\n\n", 
	     o->numeric_preconds_comp[i]);
      exit( 1 );
    }
    print_ExpNode( o->numeric_preconds_lh[i] );
    print_ExpNode( o->numeric_preconds_rh[i] );
    printf(")\n");
  }

  m = 0;
  printf("\n\nEffects:");
  for ( e = o->effects; e; e = e->next ) {
    printf("\n\neffect %d, parameters %d", m++, e->num_vars);

    for ( i = 0; i < e->num_vars; i++ ) {
      printf("\nx%d of type ", o->num_vars + i);
      print_type( e->var_types[i] );
    }
    printf("\nConditions\n");
    for ( i = 0; i < e->num_conditions; i++ ) {
      print_Fact( &(e->conditions[i]) );
      printf("\n");
    }
    for ( i = 0; i < e->num_numeric_conditions; i++ ) {
      switch ( e->numeric_conditions_comp[i] ) {
      case LE:
	printf("(< ");
	break;
      case LEQ:
	printf("(<= ");
	break;
      case EQ:
	printf("(= ");
	break;
      case GEQ:
	printf("(>= ");
	break;
      case GE:
	printf("(> ");
	break;
      default:
	printf("\nwrong comparator of Expnodes in normeff %d\n\n", 
	       e->numeric_conditions_comp[i]);
	exit( 1 );
      }
      print_ExpNode( e->numeric_conditions_lh[i] );
      print_ExpNode( e->numeric_conditions_rh[i] );
      printf(")\n");
    }

    printf("\nAdds\n");
    for ( i = 0; i < e->num_adds; i++ ) {
      print_Fact( &(e->adds[i]) );
      printf("\n");
    }
    printf("\nDels\n");
    for ( i = 0; i < e->num_dels; i++ ) {
      print_Fact( &(e->dels[i]) );
      printf("\n");
    }
    for ( i = 0; i < e->num_numeric_effects; i++ ) {
      switch ( e->numeric_effects_neft[i] ) {
      case ASSIGN:
	printf("\nassign ");
	break;
      case SCALE_UP:
	printf("\nscale-up ");
	break;
      case SCALE_DOWN:
	printf("\nscale-down ");
	break;
      case INCREASE:
	printf("\nincrease ");
	break;
      case DECREASE:
	printf("\ndecrease ");
	break;
      default:
	printf("\n\nprint normop: illegal neft %d\n\n", 
	       e->numeric_effects_neft[i]);
	exit( 1 );
      }
      print_Fluent( &(e->numeric_effects_fluent[i]) );
      print_ExpNode( e->numeric_effects_rh[i] );
    }
  }

}



void print_MixedOperator( MixedOperator *o )

{

  int i, m;
  Effect *e;
  NumericEffect *ne;
  Literal *l;

  printf("\n\n----------------Operator %s, mixed form--------------\n", 
	 o->operator->name);
 
  for ( i = 0; i < o->operator->num_vars; i++ ) {
    printf("\nx%d = %s of type ", i, gconstants[o->inst_table[i]]);
    print_type( o->operator->var_types[i] );
  }

  printf("\nPreconds:\n");
  for ( i = 0; i < o->num_preconds; i++ ) {
    print_Fact( &(o->preconds[i]) );
    printf("\n");
  }
  for ( i = 0; i < o->num_numeric_preconds; i++ ) {
    switch ( o->numeric_preconds_comp[i] ) {
    case LE:
      printf("(< ");
      break;
    case LEQ:
      printf("(<= ");
      break;
    case EQ:
      printf("(= ");
      break;
    case GEQ:
      printf("(>= ");
      break;
    case GE:
      printf("(> ");
      break;
    default:
      printf("\nwrong comparator of Expnodes in mixedpre %d\n\n", 
	     o->numeric_preconds_comp[i]);
      exit( 1 );
    }
    print_ExpNode( o->numeric_preconds_lh[i] );
    print_ExpNode( o->numeric_preconds_rh[i] );
    printf(")\n");
  }

  m = 0;
  printf("\n\nEffects:");
  for ( e = o->effects; e; e = e->next ) {
    printf("\n\neffect %d, parameters %d", m++, e->num_vars);

    for ( i = 0; i < e->num_vars; i++ ) {
      printf("\nx%d of type %s",
	     o->operator->num_vars + i, gtype_names[e->var_types[i]]);
    }
    printf("\nConditions\n");
    print_Wff( e->conditions, 0 );
    printf("\nEffect Literals");
    for ( l = e->effects; l; l = l->next ) {
      if ( l->negated ) {
	printf("\nNOT ");
      } else {
	printf("\n");
      }
      print_Fact( &(l->fact) );
    }
    printf("\nNumeric Effects");
    for ( ne = e->numeric_effects; ne; ne = ne->next ) {
      switch ( ne->neft ) {
      case ASSIGN:
	printf("\nassign ");
	break;
      case SCALE_UP:
	printf("\nscale-up ");
	break;
      case SCALE_DOWN:
	printf("\nscale-down ");
	break;
      case INCREASE:
	printf("\nincrease ");
	break;
      case DECREASE:
	printf("\ndecrease ");
	break;
      default:
	printf("\n\nprint effect: illegal neft %d\n\n", ne->neft);
	exit( 1 );
      }
      print_Fluent( &(ne->fluent) );
      print_ExpNode( ne->rh );
    }
  }

}



void print_PseudoAction( PseudoAction *o )

{

  PseudoActionEffect *e;
  int i, m;

  printf("\n\n----------------Pseudo Action %s--------------\n", 
	 o->operator->name);

  for ( i = 0; i < o->operator->num_vars; i++ ) {
    printf("\nx%d = %s of type ", i, gconstants[o->inst_table[i]]);
    print_type( o->operator->var_types[i] );
  }

  printf("\nPreconds:\n");
  for ( i = 0; i < o->num_preconds; i++ ) {
    print_Fact( &(o->preconds[i]) );
    printf("\n");
  }
  for ( i = 0; i < o->num_numeric_preconds; i++ ) {
    switch ( o->numeric_preconds_comp[i] ) {
    case LE:
      printf("(< ");
      break;
    case LEQ:
      printf("(<= ");
      break;
    case EQ:
      printf("(= ");
      break;
    case GEQ:
      printf("(>= ");
      break;
    case GE:
      printf("(> ");
      break;
    default:
      printf("\nwrong comparator of Expnodes in mixedpre %d\n\n", 
	     o->numeric_preconds_comp[i]);
      exit( 1 );
    }
    print_ExpNode( o->numeric_preconds_lh[i] );
    print_ExpNode( o->numeric_preconds_rh[i] );
    printf(")\n");
  }

  m = 0;
  printf("\n\nEffects:");
  for ( e = o->effects; e; e = e->next ) {
    printf("\n\neffect %d", m++);
    printf("\n\nConditions\n");
    for ( i = 0; i < e->num_conditions; i++ ) {
      print_Fact( &(e->conditions[i]) );
      printf("\n");
    }
    for ( i = 0; i < e->num_numeric_conditions; i++ ) {
      switch ( e->numeric_conditions_comp[i] ) {
      case LE:
	printf("(< ");
	break;
      case LEQ:
	printf("(<= ");
	break;
      case EQ:
	printf("(= ");
	break;
      case GEQ:
	printf("(>= ");
	break;
      case GE:
	printf("(> ");
	break;
      default:
	printf("\nwrong comparator of Expnodes in normeff %d\n\n", 
	       e->numeric_conditions_comp[i]);
	exit( 1 );
      }
      print_ExpNode( e->numeric_conditions_lh[i] );
      print_ExpNode( e->numeric_conditions_rh[i] );
      printf(")\n");
    }

    printf("\nAdds\n");
    for ( i = 0; i < e->num_adds; i++ ) {
      print_Fact( &(e->adds[i]) );
      printf("\n");
    }
    printf("\nDels\n");
    for ( i = 0; i < e->num_dels; i++ ) {
      print_Fact( &(e->dels[i]) );
      printf("\n");
    }
    for ( i = 0; i < e->num_numeric_effects; i++ ) {
      switch ( e->numeric_effects_neft[i] ) {
      case ASSIGN:
	printf("\nassign ");
	break;
      case SCALE_UP:
	printf("\nscale-up ");
	break;
      case SCALE_DOWN:
	printf("\nscale-down ");
	break;
      case INCREASE:
	printf("\nincrease ");
	break;
      case DECREASE:
	printf("\ndecrease ");
	break;
      default:
	printf("\n\nprint normop: illegal neft %d\n\n", 
	       e->numeric_effects_neft[i]);
	exit( 1 );
      }
      print_Fluent( &(e->numeric_effects_fluent[i]) );
      print_ExpNode( e->numeric_effects_rh[i] );
    }
  }

}



void print_Action( Action *a )

{

  ActionEffect *e;
  int i, j;

  if ( !a->norm_operator &&
       !a->pseudo_action ) {
    printf("\n\nAction REACH-GOAL");
  } else {
    printf("\n\nAction %s", a->name ); 
    for ( i = 0; i < a->num_name_vars; i++ ) {
      printf(" %s", gconstants[a->name_inst_table[i]]);
    }
  }

  printf("\n\nPreconds:\n");
  for ( i = 0; i < a->num_preconds; i++ ) {
    print_ft_name( a->preconds[i] );
    printf("\n");
  }
  for ( i = 0; i < a->num_numeric_preconds; i++ ) {
    switch ( a->numeric_preconds_comp[i] ) {
    case LE:
      printf("(< ");
      break;
    case LEQ:
      printf("(<= ");
      break;
    case EQ:
      printf("(= ");
      break;
    case GEQ:
      printf("(>= ");
      break;
    case GE:
      printf("(> ");
      break;
    default:
      printf("\nwrong comparator of Expnodes in actionpre %d\n\n", 
	     a->numeric_preconds_comp[i]);
      exit( 1 );
    }
    print_ExpNode( a->numeric_preconds_lh[i] );
    print_ExpNode( a->numeric_preconds_rh[i] );
    printf(")\n");
  }

  printf("\n\nEffects:");
  for ( j = 0; j < a->num_effects; j++ ) {
    printf("\n\neffect %d", j);
    e = &(a->effects[j]);
    if ( e->illegal ) printf(" ILLEGAL EFFECT!");
    printf("\n\nConditions\n");
    for ( i = 0; i < e->num_conditions; i++ ) {
      print_ft_name( e->conditions[i] );
      printf("\n");
    }
    for ( i = 0; i < e->num_numeric_conditions; i++ ) {
      switch ( e->numeric_conditions_comp[i] ) {
      case LE:
	printf("(< ");
	break;
      case LEQ:
	printf("(<= ");
	break;
      case EQ:
	printf("(= ");
	break;
      case GEQ:
	printf("(>= ");
	break;
      case GE:
	printf("(> ");
	break;
      default:
	printf("\nwrong comparator of Expnodes in normeff %d\n\n", 
	       e->numeric_conditions_comp[i]);
	exit( 1 );
      }
      print_ExpNode( e->numeric_conditions_lh[i] );
      print_ExpNode( e->numeric_conditions_rh[i] );
      printf(")\n");
    }
    printf("\nAdds\n");
    for ( i = 0; i < e->num_adds; i++ ) {
      print_ft_name( e->adds[i] );
      printf("\n");
    }
    printf("\nDels\n");
    for ( i = 0; i < e->num_dels; i++ ) {
      print_ft_name( e->dels[i] );
      printf("\n");
    }
    for ( i = 0; i < e->num_numeric_effects; i++ ) {
      switch ( e->numeric_effects_neft[i] ) {
      case ASSIGN:
	printf("\nassign ");
	break;
      case SCALE_UP:
	printf("\nscale-up ");
	break;
      case SCALE_DOWN:
	printf("\nscale-down ");
	break;
      case INCREASE:
	printf("\nincrease ");
	break;
      case DECREASE:
	printf("\ndecrease ");
	break;
      default:
	printf("\n\nprint normop: illegal neft %d\n\n", 
	       e->numeric_effects_neft[i]);
	exit( 1 );
      }
      if ( e->numeric_effects_fl[i] >= 0 ) {
	print_fl_name( e->numeric_effects_fl[i] );
      } else {
	printf("[UNDEF]");
      }
      print_ExpNode( e->numeric_effects_rh[i] );
    }
  }

}



void print_Action_name( Action *a )

{

  int i;

  if ( !a->norm_operator &&
       !a->pseudo_action ) {
    printf("REACH-GOAL");
  } else {
    printf("%s", a->name ); 
    for ( i = 0; i < a->num_name_vars; i++ ) {
      printf(" %s", gconstants[a->name_inst_table[i]]);
    }
  }

}



void print_lnf_Action( Action *a )

{

  ActionEffect *e;
  int i, j;

  if ( !a->norm_operator &&
       !a->pseudo_action ) {
    printf("\n\nAction REACH-GOAL");
  } else {
    printf("\n\nAction %s", a->name ); 
    for ( i = 0; i < a->num_name_vars; i++ ) {
      printf(" %s", gconstants[a->name_inst_table[i]]);
    }
  }

  printf("\n\nPreconds:\n");
  for ( i = 0; i < a->num_preconds; i++ ) {
    print_ft_name( a->preconds[i] );
    printf("\n");
  }
  for ( i = 0; i < a->num_lnf_preconds; i++ ) {
    switch ( a->lnf_preconds_comp[i] ) {
    case GEQ:
      printf("(>= ");
      break;
    case GE:
      printf("(> ");
      break;
    default:
      printf("\nwrong comparator of Expnodes in lnf actionpre %d\n\n", 
	     a->lnf_preconds_comp[i]);
      exit( 1 );
    }
    print_LnfExpNode( a->lnf_preconds_lh[i] );
    printf(" %.2f)\n", a->lnf_preconds_rh[i]);
  }

  printf("\n\nEffects:");
  for ( j = 0; j < a->num_effects; j++ ) {
    printf("\n\neffect %d COST %f", j, a->effects[j].cost);
    e = &(a->effects[j]);
    if ( e->illegal ) printf(" ILLEGAL EFFECT!");
    if ( e->removed ) printf(" REMOVED!!!");
    printf("\n\nConditions\n");
    for ( i = 0; i < e->num_conditions; i++ ) {
      print_ft_name( e->conditions[i] );
      printf("\n");
    }
    for ( i = 0; i < e->num_lnf_conditions; i++ ) {
      switch ( e->lnf_conditions_comp[i] ) {
      case GEQ:
	printf("(>= ");
	break;
      case GE:
	printf("(> ");
	break;
      default:
	printf("\nwrong comparator of Expnodes in lnf normeff %d\n\n", 
	       e->lnf_conditions_comp[i]);
	exit( 1 );
      }
      print_LnfExpNode( e->lnf_conditions_lh[i] );
      printf(" %.2f)\n", e->lnf_conditions_rh[i] );
    }
    printf("\nAdds\n");
    for ( i = 0; i < e->num_adds; i++ ) {
      print_ft_name( e->adds[i] );
      printf("\n");
    }
    printf("\nDels\n");
    for ( i = 0; i < e->num_dels; i++ ) {
      print_ft_name( e->dels[i] );
      printf("\n");
    }
    for ( i = 0; i < e->num_lnf_effects; i++ ) {
      switch ( e->lnf_effects_neft[i] ) {
      case ASSIGN:
	printf("\nassign ");
	break;
      case INCREASE:
	printf("\nincrease ");
	break;
      default:
	printf("\n\nprint lnf normop: illegal neft %d\n\n", 
	       e->lnf_effects_neft[i]);
	exit( 1 );
      }
      if ( e->lnf_effects_fl[i] >= 0 ) {
	print_fl_name( e->lnf_effects_fl[i] );
      } else {
	printf("[UNDEF]");
      }
      print_LnfExpNode( e->lnf_effects_rh[i] );
    }
  }

}



void print_type( int t )

{

  int j;

  if ( gpredicate_to_type[t] == -1 ) {
    if ( gnum_intersected_types[t] == -1 ) {
      printf("%s", gtype_names[t]);
    } else {
      printf("INTERSECTED TYPE (");
      for ( j = 0; j < gnum_intersected_types[t]; j++ ) {
	if ( gpredicate_to_type[gintersected_types[t][j]] == -1 ) {
	  printf("%s", gtype_names[gintersected_types[t][j]]);
	} else {
	  printf("UNARY INERTIA TYPE (%s)", 
		 gpredicates[gpredicate_to_type[gintersected_types[t][j]]]);
	}
	if ( j < gnum_intersected_types[t] - 1 ) {
	  printf(" and ");
	}
      }
      printf(")");
    }
  } else {
    printf("UNARY INERTIA TYPE (%s)", gpredicates[gpredicate_to_type[t]]);
  }

}



void print_Fact( Fact *f )

{

  int j;

  if ( f->predicate == -3 ) {
    printf("GOAL-REACHED");
    return;
  }

  if ( f->predicate == -1 ) {
    printf("(=");
    for ( j=0; j<2; j++ ) {
      printf(" ");
      if ( f->args[j] >= 0 ) {
	printf("%s", gconstants[(f->args)[j]]);
      } else {
	printf("x%d", DECODE_VAR( f->args[j] ));
      }
    }
    printf(")");
    return;
  }

  if ( f->predicate == -2 ) {
    printf("(!=");
    for ( j=0; j<2; j++ ) {
      printf(" ");
      if ( f->args[j] >= 0 ) {
	printf("%s", gconstants[(f->args)[j]]);
      } else {
	printf("x%d", DECODE_VAR( f->args[j] ));
      }
    }
    printf(")");
    return;
  }
    
  printf("(%s", gpredicates[f->predicate]);
  for ( j=0; j<garity[f->predicate]; j++ ) {
    printf(" ");
    if ( f->args[j] >= 0 ) {
      printf("%s", gconstants[(f->args)[j]]);
    } else {
      printf("x%d", DECODE_VAR( f->args[j] ));
    }
  }
  printf(")");

}



void print_Fluent( Fluent *f )

{

  int j, ff = f->function;

  printf("(%s", gfunctions[ff]);
  for ( j=0; j<gf_arity[ff]; j++ ) {
    printf(" ");
    if ( f->args[j] >= 0 ) {
      printf("%s", gconstants[(f->args)[j]]);
    } else {
      printf("x%d", DECODE_VAR( f->args[j] ));
    }
  }
  printf(")");

}



void print_ft_name( int index )

{

  print_Fact( &(grelevant_facts[index]) );

}



void print_fl_name( int index )

{

  int i;

  if ( index < 0 ) {
    if ( index != -2 ) {
      printf("[UNDEF]");
    } else {
      printf("[TOTAL-TIME]");
    }
    return;
  }

  if ( grelevant_fluents_lnf[index] == NULL ) {
    /* this is a non-artificial "atomic" one
     * (or the mirrored version of one)
     */
    printf("[RF%d](%s)", index, grelevant_fluents_name[index]);
  } else {
    /* this only summarizes a LNF requirement
     */
    printf("[artRF%d]", index);
    for ( i = 0; i < grelevant_fluents_lnf[index]->num_pF; i++ ) {
      printf("%.2f*", grelevant_fluents_lnf[index]->pC[i] );
      print_fl_name( grelevant_fluents_lnf[index]->pF[i] );
      if ( i < grelevant_fluents_lnf[index]->num_pF - 1 ) {
	printf(" + ");
      }
    }
  }

}



void print_LnfExpNode( LnfExpNode *n )

{

  int i;

  printf("((");
  for ( i = 0; i < n->num_pF; i++ ) {
    printf("%.2f*", n->pC[i]);
    print_fl_name( n->pF[i] );
  }
  printf(") - (");
  for ( i = 0; i < n->num_nF; i++ ) {
    printf("%.2f*", n->nC[i]);
    print_fl_name( n->nF[i] );
  }
  printf(") + %.2f)", n->c);

}



void print_op_name( int index )

{

  int i;
  Action *a = gop_conn[index].action;

  if ( !a->norm_operator &&
       !a->pseudo_action ) {
    printf("REACH-GOAL");
  } else {
    printf("%s", a->name ); 
    for ( i = 0; i < a->num_name_vars; i++ ) {
      printf(" %s", gconstants[a->name_inst_table[i]]);
    }
  }

}



void print_State( State S )

{

  int i;
  
  for ( i = 0; i < S.num_F; i++ ) {
    printf("\n");
    print_ft_name( S.F[i] );
  }
  for ( i = 0; i < gnum_relevant_fluents; i++ ) {
    printf("\n");
    print_fl_name( i );
    printf(": ");
    if ( S.f_D[i] ) {
      printf("%.2f", S.f_V[i]);
    } else {
      printf("UNDEF");
    }
  }

}








/*
 * program output routines
 */









void print_plan( void )

{  

  int i;
  float cost = 0;

  printf("\n\nff: found legal plan as follows");
  printf("\nstep ");
  for ( i = 0; i < gnum_plan_ops; i++ ) {
    printf("%4d: ", i);
    print_op_name( gplan_ops[i] );
    if ( i < gnum_plan_ops-1 ) {
      printf("\n     ");
    }
    if ( goptimization_established ) {
      cost += gop_conn[gplan_ops[i]].cost;
    }
  }
  if ( goptimization_established ) {
    printf("\nplan cost: %f", cost);
  }

}
