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
 * File: inst_final.h
 * Description: headers for final domain representation functions
 *
 *
 * Author: Joerg Hoffmann 2000
 *
 *********************************************************************/ 







#ifndef _INST_FINAL_H
#define _INST_FINAL_H



void perform_reachability_analysis( void );
int fact_adress( void );
void make_name_inst_table_from_NormOperator( Action *a, NormOperator *o, EasyTemplate *t );
void make_name_inst_table_from_PseudoAction( Action *a, PseudoAction *pa );



void collect_relevant_facts_and_fluents( void );
void create_final_goal_state( void );
Bool set_relevants_in_wff( WffNode **w );
Bool set_relevants_in_exp( ExpNode **n );
void create_final_initial_state( void );
void create_final_actions( void );
void instantiate_exp_by_action( ExpNode **n, Action *a );



void build_connectivity_graph( void );



void summarize_effects( void );
Bool same_condition( ActionEffect *e, ActionEffect *e_ );
Bool same_lnfs( LnfExpNode *l, LnfExpNode *r );
void merge_effects( ActionEffect *e, ActionEffect *e_ );



#endif /* _INST_FINAL_H */
